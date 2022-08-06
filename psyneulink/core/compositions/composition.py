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
     - `Composition_Constructor`
     - `Composition_Addition_Methods`
        • `Adding Components <Composition_Component_Addition_Methods>`
        • `Adding Pathways <Composition_Pathway_Addition_Methods>`
     - `Composition_Add_Nested`
  * `Composition_Structure`
     - `Composition_Graph`
     - `Composition_Nodes`
     - `Composition_Nested`
        • `Probes <Composition_Probes>`
     - `Composition_CIMs`
     - `Composition_Projections`
     - `Composition_Pathways`
  * `Composition_Controller`
     - `Composition_Controller_Assignment`
     - `Composition_Controller_Execution`
  * `Composition_Learning`
     - `Composition_Learning_Standard`
        • `Composition_Learning_Unsupervised`
        • `Composition_Learning_Supervised`
           - `Composition_Learning_Methods`
           - `Composition_Learning_Components`
           - `Composition_Learning_Execution`
     - `Composition_Learning_AutodiffComposition`
     - `Composition_Learning_UDF`
  * `Composition_Execution`
     - `Execution Methods <Composition_Execution_Methods>`
     - `Composition_Execution_Inputs`
        • `Composition_Input_Formats`
           - `Composition_Input_Dictionary`
           - `Composition_Programmatic_Inputs`
     - `Composition_Execution_Factors`
        • `Composition_Runtime_Params`
        • `Composition_Cycles_and_Feedback`
           - `Composition_Cycle`
           - `Composition_Feedback`
        • `Composition_Execution_Context`
        • `Composition_Timing`
        • `Composition_Reset`
        • `Composition_Compilation`
     - `Results, Reporting and Logging <Composition_Execution_Results_and_Reporting>`
  * `Composition_Visualization`
  * `Composition_Examples`
     - `Composition_Examples_Creation`
     - `Composition_Examples_Run`
     - `Composition_Examples_Learning`
     - `Composition_Examples_Input`
     - `Composition_Examples_Runtime_Params`
     - `Composition_Examples_Cycles_Feedback`
     - `Composition_Examples_Execution_Context`
     - `Composition_Examples_Reset`
     - `ShowGraph_Examples_Visualization`
  * `Composition_Class_Reference`

.. _Composition_Overview:

Overview
--------

    .. warning::
        As of PsyNeuLink 0.7.5, the API for using Compositions for Learning has been slightly changed!
        Please see `this link <RefactoredLearningGuide>` for more details.

Composition is the base class for objects that combine PsyNeuLink `Components <Component>` into an executable model.
It defines a common set of attributes possessed, and methods used by all Composition objects.

Composition `Nodes <Composition_Nodes>` are `Mechanisms <Mechanism>` and/or nested `Compositions <Composition>`.
`Projections <Projection>` connect pairs of Nodes. The Composition's `graph <Composition.graph>` stores the
structural relationships among the Nodes of a Composition and the Projections that connect them.  The Composition's
`scheduler <Composition.scheduler>` generates an execution queue based on these structural dependencies, allowing for
other user-specified scheduling and termination conditions to be specified.

.. _Composition_Creation:

Creating a Composition
----------------------

    - `Composition_Constructor`
    - `Composition_Addition_Methods`
    - `Composition_Add_Nested`

A Composition can be created by calling the constructor and specifying `Components <Component>` to be added, using
either arguments of the constructor and/or methods that allow Components to be added once it has been constructed.

.. hint::
    Although Components (Nodes and Projections) can be added individually to a Composition, it is often easier to use
    `Pathways <Composition_Pathways>` to construct a Composition, which in many cases can automaticially construct the
    Projections needed without having to specify those explicitly.

.. _Composition_Constructor:

*Using the Constructor*
~~~~~~~~~~~~~~~~~~~~~~~

The following arguments of the Composition's constructor can be used to add Compnents when it is constructed:

   .. _Composition_Pathways_Arg:

    - **pathways**
        adds one or more `Pathways <Composition_Pathways>` to the Composition; this is equivalent to constructing
        the Composition and then calling its `add_pathways <Composition.add_pathways>` method, and can use the
        same forms of specification as the **pathways** argument of that method (see `Pathway_Specification` for
        additonal details). If any `learning Pathways <Composition_Learning_Pathway>` are included, then the
        constructor's **disable_learning** argument can be used to disable learning on those by default (though it
        will still allow learning to occur on any other Compositions, either nested within the current one,
        or within which the current one is nested (see `Composition_Learning` for a full description).

   .. _Composition_Nodes_Arg:

    - **nodes**
        adds the specified `Nodes <Composition_Nodes>` to the Composition;  this is equivalent to constructing the
        Composition and then calling its `add_nodes <Composition.add_nodes>` method, and takes the same values as the
        **nodes** argument of that method (note that this does *not* construct `Pathways <Pathway>` for the specified
        nodes; the **pathways** arg or  `add_pathways <Composition.add_pathways>` method must be used to do so).

   .. _Composition_Projections_Arg:

    - **projections**
        adds the specified `Projections <Projection>` to the Composition;  this is equivalent to constructing the
        Composition and then calling its `add_projections <Composition.add_projections>` method, and takes the same
        values as the **projections** argument of that method.  In general, this is not neded -- default Projections
        are created for Pathways and/or Nodes added to the Composition using the methods described above; however
        it can be useful for custom configurations, including the implementation of specific Projection `matrices
         <MappingProjection.matrix>`.

   .. _Composition_Controller_Arg:

    - **controller**
       adds the specified `ControlMechanism` (typically an `OptimizationControlMechanism`) as the `controller
       <Composition.controller>` of the Composition, that can be used to simulate and optimize performance of the
       Composition. If this is specified, then the **enable_controller**, **controller_mode**,
       **controller_condition** and **retain_old_simulation_data** can be used to configure the controller's operation
       (see `Composition_Controller` for full description).

.. _Composition_Addition_Methods:

*Adding Components and Pathways*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - `Adding Components <Composition_Component_Addition_Methods>`
    - `Adding Pathways <Composition_Pathway_Addition_Methods>`

The methods used for adding individual Components and `Pathways <Composition_Pathways>` to a Composition are described
briefly below.  Examples of their their use are provided in `Composition_Examples_Creation`.

.. _Composition_Component_Addition_Methods:

The following methods can be used to add individual Components to an existing Composition:

    - `add_node <Composition.add_node>`

        adds a `Node <Composition_Nodes>` to the Composition.

    - `add_nodes <Composition.add_nodes>`

        adds mutiple `Nodes <Composition_Nodes>` to the Composition.

    - `add_projection <Composition.add_projection>`

        adds a `Projection <Projection>` between a pair of `Nodes <Composition_Nodes>` in the Composition.

    - `add_projections <Composition.add_projections>`

        adds `Projections <Projection>` between multiple pairs of `Nodes <Composition_Nodes>` in the Composition.

.. _Composition_Pathway_Addition_Methods:

These methods can be used to add `Pathways <Composition_Pathways>` to an existing Composition:

    - `add_pathways <Composition.add_pathways>`

        adds one or more Pathways to the Composition; this a convenience method, that determines the type of
        each Pathway, and calls the relevant ones of the following methods for each Pathway.

    - `add_linear_processing_pathway <Composition.add_linear_processing_pathway>`

        adds and a list of `Nodes <Composition_Nodes>` and `Projections <Projection>` to the Composition, inserting
        a default Projection between any adjacent set(s) of Nodes for which a Projection is not otherwise specified
        (see method documentation and `Pathway_Specification` for additonal details); returns the `Pathway` added to
        the Composition.

    COMMENT:
    The following set of `learning methods <Composition_Learning_Methods>` can be used to add `Pathways
        <Component_Pathway>` that implement `learning <Composition_Learning>` to an existing Composition:
    COMMENT

    - `add_linear_learning_pathway <Composition.add_linear_learning_pathway>`

        adds a list of `Nodes <Composition_Nodes>` and `Projections <Projection>` to implement a `learning pathway
        <Composition_Learning_Pathway>`, including the `learning components <Composition_Learning_Components>`
        needed to implement the algorithm specified in its **learning_function** argument;
        returns the `learning Pathway <Composition_Learning_Pathway>` added to the Composition.

    - `add_reinforcement_learning_pathway <Composition.add_reinforcement_learning_pathway>`

        adds a list of `Nodes <Composition_Nodes>` and `Projections <Projection>`, including the `learning components
        <Composition_Learning_Components>` needed to implement `reinforcement learning <Reinforcement>` in the
        specified pathway; returns the `learning Pathway <Composition_Learning_Pathway>` added to the Composition.

    - `add_td_learning_pathway <Composition.add_td_learning_pathway>`

        adds a list of `Nodes <Composition_Nodes>` and `Projections <Projection>`, including the `learning components
        <Composition_Learning_Components>` needed to implement `temporal differences <TDLearning>` method of
        reinforcement learning` in the specified pathway; returns the `learning Pathway <Composition_Learning_Pathway>`
        added to the Composition.

    - `add_backpropagation_learning_pathway <Composition.add_backpropagation_learning_pathway>`

        adds a list of `Nodes <Composition_Nodes>` and `Projections <Projection>`, including the `learning components
        <Composition_Learning_Components>` needed to implement the `backpropagation learning algorithm
        <BackPropagation>` in the specified pathway; returns the `learning Pathway <Composition_Learning_Pathway>`
        added to the Composition.

.. note::
  Only Mechanisms and Projections added to a Composition using the methods above belong to a Composition, even if
  other Mechanism and/or Projections are constructed in the same Python script.

A `Node <Composition_Nodes>` can be removed from a Composition using the `remove_node <Composition.remove_node>` method.


.. _Composition_Add_Nested:

*Adding Nested Compositions*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Composition can be used as a `Node <Composition_Nodes>` of another Composition, either in the **nodes** argument
of its consructor, in a `Pathway` specified in its **pathways** argument, or in one of the Composition's `addition
methods <Composition_Addition_Methods>`.  Projections can be specifed to and from the nested composition (or
created automatically if specified in a Pathway) just as for any other Node.


.. _Composition_Structure:

Composition Structure
---------------------

    - `Composition_Graph`
    - `Composition_Nodes`
    - `Composition_Nested`
    - `Composition_Pathways`

This section provides an overview of the structure of a Composition and its `Components <Component>`. Later sections
describe these in greater detail, and how they are used to implement various forms of Composition.

.. _Composition_Graph:

*Graph*
~~~~~~~

The structure of a Composition is a computational graph, the `Nodes <Composition_Nodes>` of which are `Mechanisms
<Mechanism>` and/or `nested Composition(s) <Composition_Nested>` that carry out computations; and the edges of which
can be thought of as Composition's `Projections <Projection>`, that transmit the computational results from one Node
to another Node (though see `below <Composition_Projections>` for a fuller description). The information about a
Composition's structure is stored in its `graph <Composition.graph>` attribute, that is a `Graph` object describing
its Nodes and the dependencies determined by its Projections.  There are no restrictions on the structure of the
graph, which can be `acyclic or cyclic <Composition_Acyclic_Cyclic>`, and/or hierarchical (i.e., contain one or more
`nested Compositions <Composition_Nested>`) as described below. A Composition's `graph <Composition.graph>` can be
displayed using the Composition's `show_graph <ShowGraph.show_graph>` method (see `ShowGraph_show_graph_Method`).

.. _Composition_Acyclic_Cyclic:

**Acyclic and Cyclic Graphs**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Projections are always directed (that is, information is transimtted in only one direction).  Therefore, if the
Projections among the Nodes of the Composition never form a loop, then it is a `directed acyclic graph (DAG)
<https://en.wikipedia.org/wiki/Acyclic_graph>`_, and the order in which its Nodes are executed can be determined by
the structure of the graph itself.  However if the Composition contains loops, then its structure is a `cyclic graph
<https://en.wikipedia.org/wiki/Cyclic_graph>`_, and how the Nodes in the loop are initialized and the order in which
they execute must be determined in order to execute the graph.  PsyNeuLink has procedures both for automatically
detecting handling such cycles, and also for allowing the user to specify how this is done (see
`Composition_Cycles_and_Feedback`).

COMMENT:
    XXX ADD FIGURE WITH DAG (FF) AND CYCLIC (RECURRENT) GRAPHS, OR POINT TO ONE BELOW
COMMENT


.. _Composition_Nodes:

*Nodes*
~~~~~~~

Every `Node <Composition_Nodes>` in a Composition's graph must be either a `Mechanism` or a `nested Composition
<Composition_Nested>`. The Nodes of a Composition's graph are listed in its `nodes <Composition.nodes>` attribute.
Each Node is assigned one or more `NodeRoles <NodeRole>` that designate its status in the graph.  Nodes are assigned
one or more `NodeRoles <NodeRole>` automatically when a Composition is constructed, and when Nodes or `Pathways
<Composition_Pathways>` are added to it or new `Projections <Projection>` are assigned to it. However, some of these
can be explicitly assigned by specifying the desired `NodeRole` in any of the following places:

.. _Composition_Node_Role_Assignment:

  * the **required_roles** argument of the Composition's `add_node <Composition.add_node>` or `add_nodes
    <Composition.add_nodes>` methods;

  * a tuple specifying the Node in the **pathways** argument of the Composition's constructor, a `Pathway`\\'s
    constructor, or in one of the methods used to add a `Pathway <Composition_Pathways>` to the Composition
    (see `Composition_Creation`);  the Node must be the first item of the tuple, and the `NodeRole` its 2nd item.

  * the **roles** argument of the `require_node_roles <Composition.require_node_roles>` called for an existing Node.

For example, by default, the `ORIGIN` Nodes of a Composition are assigned as its `INPUT` nodes (that is, ones that
receive the `external input <Composition_Execution_Inputs>` when it is `run <Composition.run>`), and similarly its
`TERMINAL` Nodes are assigned as its `OUTPUT` Nodes (the values of which are reported as the `results
<Composition.results>` of running the Composition). However, any other Nodes can be specified as the `INPUT` or
`OUTPUT` Nodes using the methods above, in addition to those assigned by default.  It is also possible to exclude some
roles from being assigned by default, using the `exclude_node_roles <Composition.exclude_node_roles>` method.  The
description of each `NodeRole` indicates whether it is modifiable using these methods.  All of the roles assigned
to a particular Node can be listed using the `get_roles_by_node <Composition.get_roles_by_node>` method, and all of the
nodes assigned a particular role can be listed using the `get_nodes_by_role <Composition.get_nodes_by_role>` method.


.. _Composition_Nested:

*Nested Compositions*
~~~~~~~~~~~~~~~~~~~~~

A nested Composition is one that is a `Node <Composition_Nodes>` within another Composition.  When the outer
Composition is `executed <Composition_Execution>`, the nested Composition is executed when its Node in the outer
is called to execute by the outer Composition's `scheduler <Composition.scheduler>`. Any depth of nesting of
Compositions withinothers is allowed.

*Projections to Nodes in a nested Composition.* Any Node within an outer Composition can send a `Projection
<Projection>` to any `INPUT <NodeRole.INPUT>` Node of any Composition that is enclosed within it (i.e., at any level
of nesting).  In addition, a `ControlMechanism` within an outer Composition can modulate the parameter (i.e.,
send a `ControlProjection` to the `ParameterPort`) of *any* `Mechanism <Mechanism>` in a Composition nested within it,
not just its `INPUT <NodeRole.INPUT>` Nodes.

*Projections from Nodes in a nested Composition.*  The nodes of an outer Composition can also *receive* Projections
from Nodes within a nested Composition.  This is true for any `OUTPUT <NodeRole.OUTPUT>` of the nested Composition,
and it is also true for any of its other Nodes if `allow_probes <Composition.allow_probes>` is True (the default);
if it is *CONTROL*, then only the `controller <Composition.controller>` of a Composition can receive Projections
from Nodes in a nested Composition that are not `OUTPUT <NodeRole.OUTPUT>` Nodes.

  .. _Composition_Probes:

* *Probes* -- Nodes that are not `OUTPUT <NodeRole.OUTPUT>` of a nested Composition, but project to ones in an
  outer Composition, are assigned `PROBE <NodeRole.PROBE>` in addition to their other `roles <NodeRole>` in the
  nested Composition.  The only difference between `PROBE <NodeRole.PROBE>` and `OUTPUT <NodeRole.OUTPUT>` Nodes
  is whether their output is included in the `output_values <Composition.output_values>` and `results
  <Composition.results>` attributes of the *outermost* Composition to which they project; this is determined by the
  `include_probes_in_output <Composition.include_probes_in_output>` attribute of the latter. If
  `include_probes_in_output <Composition.include_probes_in_output>` is False (the default), then the output of any
  `PROBE <NodeRole.PROBE>` Nodes are *not* included in the `output_values <Composition.output_values>` or `results
  <Composition.results>` for the outermost Composition to which they project (although they *are* still included
  in those attributes of the nested Compositions; see note below). In this respect, they can be thought of as
  "probing" - that is, providing access to "latent variables" of -- the nested Composition to which they belong --
  the values of which that are not otherwise reported as part of the outermost Composition's output or results. If
  `include_probes_in_output <Composition.include_probes_in_output>` is True, then any `PROBE <NodeRole.PROBE>` Nodes
  of any nested Compositions are treated the same as `OUTPUT <NodeRole.OUTPUT>` Nodes: their outputs are included in
  the `output_values <Composition.output_values>` and `results <Composition.results>` of the outermost Composition.
  `PROBE <NodeRole.PROBE>` Nodes can be visualized, along with any Projections treated differently from those of
  `OUTPUT <NodeRole.OUTPUT>` Nodes (i.e., when `include_probes_in_output <Composition.include_probes_in_output>` is
  False), using the Composition's `show_graph <ShowGraph.show_graph>` method, which displays them in their own color
  (pink by default).

      .. hint::
         `PROBE <NodeRole.PROBE>` Nodes are useful for `model-based optimization using an
         <OptimizationControlMechanism_Model_Based>`, in which the value of one or more Nodes in a nested Composition
         may need to be `monitored <OptimizationControlMechanism_Monitor_for_Control>` without being considered as
        part of the output or results of the Composition being optimized.

      .. note::
         The specification of `include_probes_in_output <Composition.include_probes_in_output>` only applies to a
         Composition that is not nested in another.  At present, specification of the attribute for nested
         Compositions is not supported:  the **include_probes_in_output** argument in the constructor
         for nested Compositions is ignored, and the attribute is automatically set to True.

            .. technical_note::
               This is because Compositions require access to the values of all of the output_CIM of any Compositions
               nested within them (see `below <Composition_Projections_to_CIMs>`).

.. _Composition_Nested_External_Input_Ports:

*Inputs for nested Compositions*.  If a nested Composition is an `INPUT` Node of all of the Compositions within
which it is nested, including the outermost one, then when the latter is `executed <Composition_Execution>`,
the `inputs specified <Composition_Execution_Inputs>` to its `execution method <Composition_Execution_Methods>` must
include the InputPorts of the nested Composition.  These can be accessed using the Composition's
`external_input_ports_of_all_input_nodes <Composition.external_input_ports_of_all_input_nodes>` attribute.

.. _Composition_Nested_Results:

*Results from nested Compositions.* If a nested Composition is an `OUTPUT` Node of all of the Compositions within
which it is nested, including the outermost one, then when the latter is `executed <Composition_Execution>`,
both the `output_values <Composition.output_values>` and `results <Composition.results>` of the nested Composition
are also included in those attributes of any intervening and the outermost Composition.  If `allow_probes
<Composition.allow_probes>` is set (which it is by default), then the Composition's `include_probes_in_output
<Composition.include_probes_in_output>` attribute determines whether their values are also included in the
`output_values <Composition.output_values>` and `results <Composition.results>` of the outermost Composition
(see `above <Composition_Probes>`).

.. _Composition_Nested_Learning:

*Learning in nested Compositions.* A nested Composition can also contain one or more `learning Pathways
<Composition_Learning_Pathway>`, however a learning Pathway may not extend from an enclosing Composition
to one nested within it or vice versa.  The learning Pathways within a nested Composition are executed
when that Composition is run, just like any other (see `Composition_Learning_Execution`).


.. _Composition_CIMs:

*CompositionInterfaceMechanisms*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every Composition has three `CompositionInterfaceMechanisms <CompositionInterfaceMechanism>`, described below,
that act as interfaces between it and the environment, or other Components if it is `nested <Composition_Nested>`
within another Composition.  The CompositionInterfaceMechanisms of a Composition are created and assigned to it
automatically when the Composition is constructed, and executed automatically when it executes (they should never
be constructed or executed on their own).

.. _Composition_input_CIM:

* `input_CIM <Composition.input_CIM>` - this is assigned an `InputPort` and `OutputPort` for every `INPUT
  <NodeRole.INPUT>` `Node <Composition_Nodes>` of the Composition to which it belongs. The InputPorts receive input
  from either the environment or a Composition within which it is nested. If the Composition is itself an
  `INPUT <NodeRole.INPUT>` Node of an enclosing Composition, then its input must be included in the `inputs
  <Composition_Execution_Inputs>` to that Composition when it is `executed <Composition_Execution>`. Every InputPort
  of an input_CIM is associated with an OutputPort that projects to a corresponding `INPUT <NodeRole.INPUT>` Node
  of the Composition.

.. _Composition_parameter_CIM:

* `parameter_CIM <Composition.parameter_CIM>` - this is assigned an `InputPort` and `OutputPort` for every
  `Parameter` of every `Node <Composition_Nodes>` of the Composition that is `modulated <ModulatorySignal_Modulation>`
  by a `ModulatoryMechanism` (usually a `ControlMechanism`) outside of the Composition (i.e., from an enclosing
  Composition within which it is `nested <Composition_Nested>`).  The InputPort receives a Projection from a
  `ModulatorySignal` on the ModulatoryMechanism, and the paired OutputPort of the parameter_CIM conveys this via
  ModulatoryProjection to the `ParameterPort` for the Paremeter of the Mechanism to be modulated.

  .. technical_note::
    The Projection from a ModulatoryMechanism to the InputPort of a parameter_CIM is the only instance in which a
    MappingProjection is used as an `efferent projection <Mechanism_Base.efferents>` of a ModulatoryMechanism.

.. _Composition_output_CIM:

* `output_CIM <Composition.output_CIM>` - this is assigned an `InputPort` and `OutputPort` for every `OUTPUT
  <NodeRole.OUTPUT>` `Node <Composition_Nodes>` of the Composition to which it belongs. Each InputPort receives input
  from an `OUTPUT <NodeRole.OUTPUT>` Node of the Composition, and its `value <InputPort.value>` is assigned as the
  `value <OutputPort.value>` of a corresponding OutputPort.  The latter are assigned to the `output_values
  <Composition.output_values>` and `results <Composition.results>` attributes of the Composition.  If the Composition
  is `nested <Composition_Nested>` within another, then the output_CIM's `output_ports <Mechanism_Base.output_ports>`
  send Projections to Components of the Composition within which it is nested.  If it is an `OUTPUT <NodeRole.OUTPUT>`
  Node of the enclosing Composition, then its OutputPorts project the `output_CIM <Composition.output_CIM>` of the
  enclosing Composition, its `output_values <Composition.output_values>` are included in those of the enclosing
  Composition.  If the Composition has an `PROBE <NodeRole.PROBE>` Nodes, then they too project to the Composition's
  output_CIM.  If the Composition is nested in another, then the `values <Mechanism_Base.value>` of the `PROBE
  <NodeRole.PROBE>` Nodes are also included in the Composition's `output_values <Composition.output_values>`;  if it
  is an outer Composition (i.e. not nested in any other), then the Compositions' `include_probes_in_output
  <Composition.include_probes_in_output>` attribute determines whether their values are included in its `output_values
  <Composition.output_values>` and `results <Composition.results>` attributes (see `Probes <Composition_Probes>` for
  additional details).


.. _Composition_Projections:

*Projections*
~~~~~~~~~~~~~

`Projections <Projection>` can be thought of as directed edges of the Composition's `graph <Composition_Graph>`,
insofar as they are always from one Node to a single other Node, and serve to convey the results of the sender's
computation as input to the receiver. However, they are not edges in the strictest senese, for two reasons:
First, they too can carry out (restricted) computations, such as matrix transformation by a `MappingProjection`.
Second, they can be the receiver of a Projection, as in the case of a MappingProjection that receives a
`LearningProjection` used to modify its `matrix <MappingProjection.matrix>` parameter.  Nevertheless, since they
define the connections and therefore dependencies among the Composition's Nodes, they determine the structure of its
graph.  Subsets of Nodes connected by Projections can be defined as a `Pathway <Pathway>` as decribed under
`Composition_Pathways` below).

.. _Composition_Graph_Projection_Vertices:
.. technical_note::
    Because Projections are not strictly edges, they are assigned to `vertices <Graph.vertices>` in the Composition's
    `graph <Composition.graph>`, along with its Nodes.  The actual edges are implicit in the dependencies determined
    by the Projections, and listed in the graph's `dependency_dict <Graph.dependency_dict>`.

Although individual Projections are directed, pairs of Nodes can be connected with Projections in each direction
(forming a local `cycle <Composition_Cycle>`), and the `AutoAssociativeProjection` class of Projection can even
connect a Node with itself.  Projections can also connect the Node(s) of a Composition to one(s) `nested within
it <Composition_Nested>`.  In general, these are to the `INPUT <NodeRole.INPUT>` Nodes and from the `OUTPUT
<NodeRole.OUTPUT>` Nodes of a `nested Composition <Composition_Nested>`, but if the Composition's `allow_probes
<Composition.allow_probes>` attribute is not False, then Projections can be received from any Nodes within a nested
Composition (see `Probes <Composition_Probes>` for additional details). A  ControlMechanism can also control (i.e.,
send a `ControlProjection`) to any Node within a nested Composition.

Projections can be specified between `Mechanisms <Mechanism>` before they are added to a Composition.  If both
Mechanisms are later added to the same Composition, and the Projection between them is legal for the Composition,
then the Projection between them is added to it and is used during its `execution <Composition_Execution>`.
However, if the Projection is not legal for the Composition (e.g., the Mechanisms are not assigned as `INTERNAL
<NodeRole.INTERNAL>` `Nodes <Composition_Nodes>` of two different `nested Compositions <Composition_Nested>`),
the Projection will still be associated with the two Mechanisms (i.e., listed in their `afferents
<Mechanism_Base.afferents>` and `efferents <Mechanism_Base.efferents>` attributes, respectively), but it is not
added to the Composition and not used during its execution.

    .. hint::
        Projections that are associated with the `Nodes <Composition_Nodes>` of a Composition but are not in the
        Composition itself (and, accordingly, *not* listed it is `projections <Composition.projections>` attribute)
        can still be visualized using the Composition's `show_graph <ShowGraph.show_graph>` method, by specifying its
        **show_projections_not_in_composition** argument as True; Projections not in the Composition appear in red.

.. technical_note::

    .. _Composition_Projections_to_CIMs:

    Although Projections can be specified to and from Nodes within a nested Composition, these are actually
    implemented as Projections to or from the nested Composition's `input_CIM <Composition.input_CIM>`,
    `parameter_CIM <Composition.parameter_CIM>` or `output_CIM <Composition.output_CIM>`, respectively;
    those, in turn, send or receive Projections to or from the specified Nodes within the nested Composition.
    `PROBE <NodeRole.PROBE>` Nodes of a nested Composition, like `OUTPUT <NodeRole.OUTPUT>` Nodes,
    project to the Node of an enclosing Composition via the nested Composition's `output_CIM
    <Composition.output_CIM>`, and those of any intervening Compositions if it is nested more than one level deep.
    The outputs of `PROBE <NodeRole.PROBE>` Nodes are included in the `output_values <Composition.output_values>` and
    `results <Composition.results>` of such intervening Compositions (since those values are derived from the
    `output_ports <Mechanism_Base.output_ports>` of the Composition's `output_CIM <Composition.output_CIM>`.
    Specifying `include_probes_in_output <Composition.include_probes_in_output>` has no effect on this behavior
    for intervening Compositions;  it only applies to the outermost Composition to which a PROBE Node projects
    (see `Probes <Composition_Probes>` for additional details).

.. _Composition_Pathways:

*Pathways*
~~~~~~~~~~

A `Pathway` is an alternating sequence of `Nodes <Composition_Nodes>` and `Projections <Projection>` in a Composition.
Although a Composition is not required to have any Pathways, these are useful for constructing Compositions, and are
required for implementing `learning <Composition_Learning>` in a Composition. Pathways can be specified in the
**pathways** argument of the Composition's constructor, or using one of its `Pathway addition methods
<Composition_Pathway_Addition_Methods>`.  Pathways must be linear (that is, the cannot have branches), but they can be
continguous, overlapping, intersecting, or disjoint, and can have one degree of converging and/or diverging branches
(meaning that their branches can't branch). Each Pathway has a name (that can be assigned when it is constructed) and
a set of attributes, including a `pathway <Pathway.pathway>` attribute that lists the Nodes and Projections in the
Pathway, a `roles <Pathway.roles>` attribute that lists the `PathwayRoles <PathwayRole>` assigned to it (based on
the `NodeRoles <NodeRole>` assigned to its Nodes), and attributes for particular types of nodes (e.g., `INPUT` and
`OUTPUT`) if the Pathway includes nodes assigned the corresponding `NodeRoles <NodeRole>`. If a Pathway does not have
a particular type of Node, then its attribute returns None. There are
COMMENT:
ADD modulatory Pathways
three types of Pathways: processing Pathways, `control Pathways <Composition_Control_Pathways>`, and `learning Pathways
<Composition_Learning_Pathway>`.  Processing Pathways are ones not configured for control or learning.  The latter
two types are described in the sections on `Composition_Control` and `Composition_Learning`, respectively.  All of the
Pathways in a Composition are listed in its `pathways <Composition.pathways>` attribute.
COMMENT
two types of Pathways: processing Pathways and `learning Pathways <Composition_Learning_Pathway>`.  Processing
Pathways are ones not configured for learning; learning Pathways are described under `Composition_Learning`. All
of the Pathways in a Composition are listed in its `pathways <Composition.pathways>` attribute.


.. _Composition_Controller:

Controlling a Composition
-------------------------

      - `Composition_Controller_Assignment`
      - `Composition_Controller_Execution`


A Composition can be assigned a `controller <Composition.controller>`.  This must be an `OptimizationControlMechanism`,
or a subclass of one, that modulates the parameters of Components within the Composition (including Components of
nested Compositions). It typically does this based on the output of an `ObjectiveMechanism` that evaluates the value
of other Mechanisms in the Composition, and provides the result to the `controller <Composition.controller>`.

.. _Composition_Controller_Assignment:

*Assigning a Controller*
~~~~~~~~~~~~~~~~~~~~~~~~

A `controller <Composition.controller>` can be assigned either by specifying it in the **controller** argument of the
Composition's constructor, or using its `add_controller <Composition.add_controller>` method.
COMMENT:
The Node is assigned the `NodeRole` `CONTROLLER`.
COMMENT

COMMENT:
TBI FOR COMPOSITION˚
CONTROLLER CAN BE SPECIFIED BY AS True, BY CLASS (E.G., OCM), OR CONSTRUCTOR
IF TRUE, CLASS OR BY CONSTRUCTOR WITHOUT OBJECTIVE_MECHANISM SPEC, A DEFAULT IS OBJ_MECH IS CREATED
IF A DEFAULT OBJ MECH IS CREATED, OR NEITHER OBJ_MECH NOR OCM HAVE MONITOR FOR CONTROL SPECIFIED, THEN
PRIMARY OUTPUTPORT OF ALL OUTPUT NODES OF COMP ARE USED (MODULO SPEC ON INDIVIDUAL MECHS)

Specifying Parameters to Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

*Controller Execution*
~~~~~~~~~~~~~~~~~~~~~~

The `controller <Composition.controller>` is executed only if the Composition's `enable_controller
<Composition.enable_controller>` attribute is True.  This is generally done automatically when the controller is
is `assigned <Composition_Controller_Assignment>`.  If `enabled <Composition.enable_controller>`, the controller is
executed either before or after all of the other Components in the Composition have been executed at a given
`TimeScale`, and if its specified `Condition <Composition.controller_condition>` has been met, as determined by the
Composition's `controller_mode <Composition.controller_mode>`, `controller_time_scale
<Composition.controller_time_scale>` and `controller_condition <Composition.controller_condition>` attributes. By
default, a controller is enabled, and executes after the rest of the Composition (`controller_mode
<Composition.controller_mode>`\\= *AFTER*) at the end of every trial (`controller_time_scale
<Composition.controller_time_scale>`\\= `TimeScale.TRIAL` and `controller_condition <Composition.controller_condition>`
= `Always()`). However, `controller_mode <Composition.controller_mode>` can be used to specify execution of the
controller before the Composition; `controller_time_scale <Composition.controller_time_scale>` can be used to specify
execution at a particular `TimeScale` (that is at the beginning or end of every `TIME_STEP <TimeScale.TIME_STEP>`,
`PASS <TimeScale._PASS>, or `RUN <TimeScale.RUN>`); and `controller_condition <Composition.controller_condition>` can
be used to specify a particular `Condition` that must be satisified for the controller to execute.  Arguments for all
three of these attributes can be specified in the Composition's constructor, or programmatically after it is
constructed by assigning the desired value to the corresponding attribute.

.. _Composition_Learning:

Learning in a Composition
-------------------------

Learning is used to modify the `Projections <Projection>` between Mechanisms in a Composition.  More specifically,
it modifies the `matrix <MappingProjection.matrix>` parameter of the `MappingProjections <MappingProjection>` within a
`learning Pathway <Composition_Learning_Pathway>`, which implements the conection weights (i.e., strengths of
associations between representations in the Mechanisms) within a `Pathway`.  If learning is implemented for a
Composition, it can be executed calling the Composition's `learn <Composition.learn>` method (see
`Composition_Learning_Execution` and `Composition_Execution` for additional details).


.. _Composition_Learning_Configurations:

*Configuring Learning in a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are three ways of configuring learning in a Composition:

i) using `standard PsyNeuLink Components <Composition_Learning_Standard>`

ii) using the `AutodiffComposition <Composition_Learning_AutodiffComposition>` -- a specialized subclass of Composition
    that executes learning using `PyTorch <https://pytorch.org>`_

iii) using `UserDefinedFunctions <UserDefinedFunction>`.

The advantage of using standard PsyNeuLink compoments is that it assigns each operation involved in learning to a
dedicated Component. This helps make clear exactly what those operations are, the sequence in which they are carried
out, and how they interact with one another.  However, this can also make execution inefficient, due to the overhead
incurred by distributing the calculations over different Components.  If more efficient computation is critical,
then the `AutodiffComposition` can be used to execute a compatible PsyNeuLink Composition in PyTorch, or one or more
`UserDefinedFunctions <UserDefinedFunction>` can be assigned to either PyTorch functions or those in any other Python
environment that implements learning and accepts and returns tensors. Each of these approaches is described in more
detail below.

.. _Composition_Learning_Standard:

*Learning Using PsyNeuLink Components*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Composition_Learning_Unsupervised`
* `Composition_Learning_Supervised`

When learning is implemented using standard PsyNeuLink Components, each calculation and/or operation involved in
learning -- including those responsible for computing errors, and for using those errors to modify the Projections
between Mechanisms, is assigned to a different PsyNeuLink `learning-related Component
<Composition_Learning_Components>`.  These can be used to implement all types of learning.  Learning is generally
considered to fall into two broad classes:  *unsupervised*, in which connections weights are modified
by mere exposure to the inputs in order to capture structure and/or relationships among them;  and *supervised*,
which in which the connection weights are modified so that each input generates a desired output (see
`<https://www.geeksforgeeks.org/supervised-unsupervised-learning/>`_ for a useful summary).  Both types of
learning can be implemented in a Composition, using `LearningMechanisms <LearningMechanism>` that compute the
changes to make to the `matrix <MappingProjection.matrix>` parameter of `MappingProjections <MappingProjection>`
being learned, and `LearningProjections <LearningProjection>` that apply those changes to those MappingProjections.
In addition, supervised learning uses an `ObjectiveMechanism` -- usually a `ComparatorMechanism` -- to compute the error
between the response generated by the last Mechanism in a `learning Pathway <Composition_Learning_Pathway>` (to the
input provided to the first Mechanism in the `Pathway`) and the target stimulus used to specify the desired response.
In most cases, the LearningMechanisms, LearningProjections and, where needed, ObjectiveMechanism are generated
automatically, as described for each type of learning below.  However, these can also be configured manually using
their constructors, or modified by assigning values to their attributes.

.. _Composition_Learning_Unsupervised:

Unsupervised Learning
^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^
* `Composition_Learning_Methods`
* `Composition_Learning_Components`
* `Composition_Learning_Execution`

COMMENT:
    TBI:  Supervised learning is implemented using a Composition's `add_learning_pathway` method, and specifying an
    appropriate `LearningFunction <LearningFunctions>` in its **learning_function** argument.
    XXXMORE HERE ABOUT TYPES OF FUNCTIONS
    • MODIFY REFERENCE TO LEARNING COMPONENT NAMES WHEN THEY ARE IMPLEMENTED AS AN ENUM CLASS
    • ADD EXAMPLES - POINT TO ONES IN BasicsAndPrimer
COMMENT

.. _Composition_Learning_Methods:

*Supervised Learning Methods*
=============================

Supervised learning is implemented in a Composition by specifying a `learning Pathway <Composition_Learning_Pathway>`
in the **pathways** argumemt of the Composition's constructor, its `add_pathways <Composition.add_pathways>` method,
or one of its learning methods.  If the constructor or `add_pathways <Composition.add_pathways>` method is used,
then the `Pathway specification <Pathway_Specification>` must be the first item in a tuple, followed by a
`LearningFunction` as its 2nd item that specfies the type of learning.  Alternatively, a `learning Pathway
<Composition_Learning_Pathway>` can be added to a Composition by specifying the `Pathway` to be learned in the one
of the Composition's learning methods, of which there are currently three:

    • `add_reinforcement_learning_pathway` -- uses `Reinforcement`;
    • `add_td_learning_pathway` -- uses `TDLearning`;
    • `add_backpropagation_learning_pathway` -- uses `BackPropagation`.

Each uses the Composition's `add_linear_processing_pathway` method to create a `learning Pathway
<Composition_Learning_Pathway>` using the corresponding `LearningFunction`.

.. _Composition_Learning_Pathway:

*Supervised Learning Pathways*
==============================

A *learning pathway* is a contiguous sequence of `ProcessingMechanisms <ProcessingMechanism>` and the
`MappingProjections <MappingProjection>` between them, in which supervised learning is used to modify the `matrix
<MappingProjection.matrix>` parameter of the `MappingProjections <MappingProjection>` in the sequence, so that the
input to the first ProcessingMechanism in the sequence generates an output from the last ProcessingMechanism that
matches as closely as possible a target value `specified as input <Composition_Target_Inputs>` in the Composition's
`learn <Composition.learn>` method. The Mechanisms in the pathway must be compatible with learning (that is, their
`function <Mechanism_Base.function>` must be compatible with the `function <LearningMechanism.function>` of the
`LearningMechanism` for the MappingProjections they receive (see `LearningMechanism_Function`).  The Composition's
`learning methods <Composition_Learning_Methods>` return a learning `Pathway`, in which its `learning_components
<Pathway.learning_components>` attribute is assigned a dict containing the set of learning components generated for
the Pathway, as described below.

.. _Composition_Learning_Components:

*Supervised Learning Components*
================================

For each `learning pathway <Composition_Learning_Pathway>` specified in the **pathways** argument of a Composition's
constructor or one of its `learning methods <Composition_Learning_Methods>`, it creates the following Components,
and assigns to them the `NodeRoles <NodeRole>` indicated:

    .. _TARGET_MECHANISM:
    * *TARGET_MECHANISM* -- receives the desired `value <Mechanism_Base.value>` for the `OUTPUT_MECHANISM`, that is
      used by the *OBJECTIVE_MECHANISM* as the target in computing the error signal (see above);  that value must be
      specified as an input to the TARGET_MECHANISM, either in the **inputs** argument of the Composition's `learn
      <Composition.learn>` method, or in its **targets** argument in an entry for either the *TARGET_MECHANISM* or
      the `OUTPUT_MECHANISM <OUTPUT_MECHANISM>` (see `below <Composition_Target_Inputs>`); the Mechanism is assigned
      the `NodeRoles <NodeRole>` `TARGET` and `LEARNING` in the Composition.
    ..
    * a MappingProjection that projects from the *TARGET_MECHANISM* to the *TARGET* `InputPort
      <ComparatorMechanism_Structure>` of the *OBJECTIVE_MECHANISM*.
    ..
    * a MappingProjection that projects from the last ProcessingMechanism in the learning Pathway to the *SAMPLE*
      `InputPort  <ComparatorMechanism_Structure>` of the *OBJECTIVE_MECHANISM*.
    ..
    .. _OBJECTIVE_MECHANISM:
    * *OBJECTIVE_MECHANISM* -- usually a `ComparatorMechanism`, used to `calculate an error signal
      <ComparatorMechanism_Execution>` for the sequence by comparing the value received by the ComparatorMechanism's
      *SAMPLE* `InputPort <ComparatorMechanism_Structure>` (from the `output <LearningMechanism_Activation_Output>` of
      the last Processing Mechanism in the `learning Pathway <Composition_Learning_Pathway>`) with the value received
      in the *OBJECTIVE_MECHANISM*'s *TARGET* `InputPort <ComparatorMechanism_Structure>` (from the *TARGET_MECHANISM*
      generated by the method -- see below); this is assigned the `NodeRole` `LEARNING` in the Composition.
    ..
    .. _LEARNING_MECHANISMS:
    * *LEARNING_MECHANISMS* -- a `LearningMechanism` for each MappingProjection in the sequence, each of which
      calculates the `learning_signal <LearningMechanism.learning_signal>` used to modify the `matrix
      <MappingProjection.matrix>` parameter for the coresponding MappingProjection, along with a `LearningSignal` and
      `LearningProjection` that convey the `learning_signal <LearningMechanism.learning_signal>` to the
      MappingProjection's *MATRIX* `ParameterPort<Mapping_Matrix_ParameterPort>`;  depending on learning method,
      additional MappingProjections may be created to and/or from the LearningMechanism -- see
      `LearningMechanism_Learning_Configurations` for details); these are assigned the `NodeRole` `LEARNING` in the
      Composition.
    ..
    .. _LEARNING_FUNCTION:
    * *LEARNING_FUNCTION* -- the `LearningFunction` used by each of the `LEARNING_MECHANISMS` in the learning pathway.
    ..
    .. _LEARNED_PROJECTIONS:
    * *LEARNED_PROJECTIONS* -- a `LearningProjection` from each `LearningMechanism` to the `MappingProjection`
      for which it modifies it s`matrix <MappingProjection.matrix>` parameter.

It also assigns the following item to the list of `learning_components` for the pathway:

    .. _OUTPUT_MECHANISM:
    * *OUTPUT_MECHANISM* -- the final `Node <Composition_Nodes>` in the learning Pathway, the target `value
      <Mechanism_Base.value>` for which is specified as input to the `TARGET_MECHANISM`; the Node is assigned
      the `NodeRoles <NodeRole>` `OUTPUT` in the Composition.

The items with names listed above are placed in a dict that is assigned to the `learning_components
<Pathway.learning_components>` attribute of the `Pathway` returned by the learning method used to create the `Pathway`;
they key for each item in the dict is the name of the item (as listed above), and the object(s) created of that type
are its value (see `LearningMechanism_Single_Layer_Learning` for a more detailed description and figure showing these
Components).

If the learning Pathway <Composition_Learning_Pathway>` involves more than two ProcessingMechanisms (e.g. using
`add_backpropagation_learning_pathway` for a multilayered neural network), then multiple LearningMechanisms are
created, along with MappingProjections that provide them with the `error_signal <LearningMechanism.error_signal>`
from the preceding LearningMechanism, and `LearningProjections <LearningProjection>` that modify the corresponding
MappingProjections (*LEARNED_PROJECTION*\\s) in the `learning Pathway <Composition_Learning_Pathway>`, as shown for
an example in the figure below. These additional learning components are listed in the *LEARNING_MECHANISMS* and
*LEARNED_PROJECTIONS* entries of the dictionary assigned to the `learning_components <Pathway.learning_components>`
attribute of the `learning Pathway <Composition_Learning_Pathway>` return by the learning method.

.. _Composition_MultilayerLearning_Figure:

**Figure: Supervised Learning Components**

.. figure:: _static/Composition_Multilayer_Learning_fig.svg
   :alt: Schematic of LearningMechanism and LearningProjections in a Process

   *Components for supervised learning Pathway*: the Pathway has three Mechanisms generated by a call to a `supervised
   learning method <Composition_Learning_Methods>` (e.g., ``add_backpropagation_learning_pathway(pathway=[A,B,C])``),
   with `NodeRole` assigned to each `Node <Composition_Nodes>` in the Composition's `graph <Composition.graph>` (in
   italics below Mechanism type) and  the names of the learning components returned by the learning method (capitalized
   and in italics, above each Mechanism).

The description above (and `example <Composition_Examples_Learning_XOR>` >below) pertain to simple linear sequences.
However, more complex configurations, with convergent, divergent and/or intersecting sequences can be built using
multiple calls to the learning method (see `example <BasicsAndPrimer_Rumelhart_Model>` in `BasicsAndPrimer`). In
each the learning method determines how the sequence to be added relates to any existing ones with which it abuts or
intersects, and automatically creates andconfigures the relevant learning components so that the error terms are
properly computed and propagated by each LearningMechanism to the next in the configuration. It is important to note
that, in doing so, the status of a Mechanism in the final configuration takes precedence over its status in any of
the individual sequences specified in the `learning methods <Composition_Learning_Methods>` when building the
Composition.  In particular, whereas ordinarily the last ProcessingMechanism of a sequence specified in a learning
method projects to a *OBJECTIVE_MECHANISM*, this may be superceded if multiple sequences are created. This is the
case if: i) the Mechanism is in a seqence that is contiguous (i.e., abuts or intersects) with others already in the
Composition, ii) the Mechanism appears in any of those other sequences and, iii) it is not the last Mechanism in
*all* of them; in that in that case, it will not project to a *OBJECTIVE_MECHANISM* (see `figure below
<Composition_Learning_Output_vs_Terminal_Figure>` for an example).  Furthermore, if it *is* the last Mechanism in all
of them (that is, all of the specified pathways converge on that Mechanism), only one *OBJECTIVE_MECHANISM* is created
for that Mechanism (i.e., not one for each sequence).  Finally, it should be noted that, by default, learning components
are *not* assigned the `NodeRole` of `OUTPUT` even though they may be the `TERMINAL` Mechanism of a Composition;
conversely, even though the last Mechanism of a `learning Pathway <Composition_Learning_Pathway>` projects to an
*OBJECTIVE_MECHANISM*, and thus is not the `TERMINAL` `Node <Composition_Nodes>` of a Composition, if it does not
project to any other Mechanisms in the Composition it is nevertheless assigned as an `OUTPUT` of the Composition. That
is, Mechanisms that would otherwise have been the `TERMINAL` Mechanism of a Composition preserve their role as an
`OUTPUT` Node of the Composition if they are part of a `learning Pathway <Composition_Learning_Pathway>` eventhough
they project to another Mechanism (the *OBJECTIVE_MECHANISM*) in the Composition.

.. _Composition_Learning_Output_vs_Terminal_Figure:

    **OUTPUT** vs. **TERMINAL** Roles in Learning Configuration

    .. figure:: _static/Composition_Learning_OUTPUT_vs_TERMINAL_fig.svg
       :alt: Schematic of Mechanisms and Projections involved in learning

       Configuration of Components generated by the creation of two intersecting `learning Pathways
       <Composition_Learning_Pathway>` (e.g., ``add_backpropagation_learning_pathway(pathway=[A,B])`` and
       ``add_backpropagation_learning_pathway(pathway=[D,B,C])``).  Mechanism B is the last Mechanism of the sequence
       specified for the first pathway, and so would project to a `ComparatorMechanism`, and would be assigned as an
       `OUTPUT` `Node <Composition_Nodes>` of the Composition, if that pathway was created on its own. However, since
       Mechanims B is also in the middle of the sequence specified for the second pathway, it does not project to a
       ComparatorMechanism, and is relegated to being an `INTERNAL` Node of the Composition Mechanism C is now the
       one that projects to the ComparatorMechanism and assigned as the `OUTPUT` Node.

.. _Composition_Learning_Execution:

*Execution of Learning*
=======================

For learning to occur when a Composition is run, its `learn <Composition.learn>` method must be used instead of the
`run <Composition.run>` method, and its `disable_learning <Composition.disable_learning>` attribute must be False.
When the `learn <Composition.learn>` method is used, all Components *unrelated* to learning are executed in the same
way as with the `run <Composition.run>` method.  If the Composition has any `nested Composition <Composition_Nested>`
that have `learning Pathways <Composition_Learning_Pathway>`, then learning also occurs on all of those for which
the `disable_learning <Composition.disable_learning>` attribute is False.  This is true even if the `disable_learning
<Composition.disable_learning>` attribute is True for which the Composition on which the  `learn <Composition.learn>`
method was called.

When a Composition is run that contains one or more `learning Pathways <Composition_Learning_Pathway>`, all of the
ProcessingMechanisms for a pathway are executed first, and then its `learning components
<Composition_Learning_Components>`.  This is shown in an animation of the XOR network from the `example above
<Composition_XOR_Example>`:

.. _Composition_Learning_Animation_Figure:

    **Composition with Learning**

    .. figure:: _images/Composition_XOR_animation.gif
       :alt: Animation of Composition with learning

       Animation of XOR Composition in example above when it is executed by calling its `learn <Composition.learn>`
       method with the argument ``animate={'show_learning':True}``.

.. note::
    Since the `learning components <Composition_Learning_Components>` are not executed until after the
    processing components, the change to the weights of the MappingProjections in a learning pathway are not
    made until after it has executed.  Thus, as with `execution of a Projection <Projection_Execution>`, those
    changes will not be observed in the values of their `matrix <MappingProjection.matrix>` parameters until after
    they are next executed (see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).

.. _Composition_Learning_AutodiffComposition:

*Learning Using AutodiffCompositon*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

COMMENT:
    Change reference to example below to point to Rumelhart Semantic Network Model Script once implemented
COMMENT

`AutodiffCompositions <AutodiffComposition>` provide the ability to execute a composition using `PyTorch
<https://pytorch.org>`_ (see `example <BasicsAndPrimer_Rumelhart_Model>` in `BasicsAndPrimer`).  The
AutodiffComposition constructor provides arguments for configuring the PyTorch implementation in various ways; the
Composition is then built using the same methods (e.g., `add_node`, `add_projection`, `add_linear_processing_pathway`,
etc.) as any other Composition. Note that there is no need to use any `learning methods <Composition_Learning_Methods>`
— AutodiffCompositions automatically creates backpropagation learning pathways <Composition_Learning_Pathway>` between
all input - output `Node <Composition_Nodes>` paths. It can be run just as a standard Composition would - using `learn
<AutodiffComposition.learn>` for learning mode, and `run <AutodiffComposition.run>` for test mode.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If execution efficiency is critical and the `AutodiffComposition` is too restrictive, a function from any Python
environment that supports learning can be assigned as the `function <Mechanism_Base.function>` of a `Mechanism
<Mechanism>`, in which case it is automatically  wrapped as `UserDefinedFunction`.  For example, the `forward and
backward methods <https://pytorch.org/docs/master/notes/extending.html>`_ of a PyTorch object can be assigned in this
way.  The advanatage of this approach is that it can be applied to any Python function that adheres to the requirements
of a `UserDefinedFunction`. It must be carefully coordinated with the execution of other learning-related Components in
the Composition, to insure that each function is called at the appropriate times during execution.  Furthermore, as
with an `AutodiffComposition`, the internal constituents of the object (e.g., intermediates layers of a neural network
model) are not accessible to other Components in the Composition (e.g., as a source of information or for modulation).


.. _Composition_Execution:

Executing a Composition
-----------------------

    - `Execution Methods <Composition_Execution_Methods>`
    - `Composition_Execution_Inputs`
        • `Composition_Input_Dictionary`
        • `Composition_Programmatic_Inputs`
    - `Composition_Execution_Factors`
        • `Composition_Runtime_Params`
        • `Composition_Cycles_and_Feedback`
        • `Composition_Execution_Context`
        • `Composition_Timing`
        • `Composition_Reset`
        • `Composition_Compilation`
    - `Results, Reporting and Logging <Composition_Execution_Results_and_Reporting>`


.. _Composition_Execution_Methods:

There are three methods for executing a Composition:

  * `run <Composition.run>` - executes one or more `TRIAL <TimeScale.TRIAL>`\\s without learning;

  * `learn <Composition.learn>` - executes one or more `TRIAL <TimeScale.TRIAL>`\\s with learning,
    if the network is configured for `learning <Composition_Learning>`.

  * `execute <Composition.execute>` - executes a single `TRIAL <TimeScale.TRIAL>` without learning.

The `run <Composition.run>` and `learn <Composition.learn>` methods are the most commonly used.  Both of these
can execute multiple trials (specified in their **num_trials** argument), calling the Composition's `execute
<Composition.execute>` method for each `TRIAL <TimeScale.TRIAL>`.  The `execute <Composition.execute>` method
can also be called directly, but this is useful mostly for debugging.

.. hint::
   Once a Composition has been constructed, it can be called directly. If it is called with no arguments, and
   has executed previously, the `result <Composition_Execution_Results>` of the last `TRIAL <TimeScale.TRIAL>`
   of execution is returned; otherwise None is returned.  If it is called with arguments, then either `run
   <Composition.run>` or `learn <Composition.learn>` is called, based on the arguments provided:  If the
   Composition has any `learning_pathways <Composition_Learning_Pathway>`, and the relevant `TARGET_MECHANISM
   <Composition_Learning_Components>`\\s are specified in the `inputs argument <Composition_Execution_Inputs>`,
   then `learn <Composition.learn>` is called;  otherwise, `run <Composition.run>` is called.  In either case,
   the return value of the corresponding method is returned.

.. _Composition_Execution_Num_Trials:

*Number of trials*. If the the `execute <Composition.execute>` method is used, a single `TRIAL <TimeScale.TRIAL>` is
executed;  if the **inputs** specifies more than one `TRIAL <TimeScale>`\\s worth of input, an error is generated.
For the `run <Composition.run>` and `learn <Composition.learn>`, the **num_trials** argument can be used to specify
an exact number of `TRIAL <TimeScale.TRIAL>`\\s to execute; if its value execeeds the number of inputs provided for
each Node in the **inputs** argument, then the inputs are recycled from the beginning of the lists, until the number
of `TRIAL <TimeScale.TRIAL>`\\s specified in **num_trials** has been executed.  If **num_trials** is not specified,
then a number of `TRIAL <TimeScale.TRIAL>`\\s is executed equal to the number of inputs provided for each `Node
<Composition_Nodes>` in **inputs** argument.

.. _Composition_Execution_Learning_Inputs:

*Learning*. If a Composition is configured for `learning <Composition_Learning>` then, for learning to occur,
its `learn <Composition.learn>` method must be used in place of the `run <Composition.run>` method, and its
`disable_learning <Composition.disable_learning>` attribute must be False (the default). A set of targets must also
be specified (see `below <Composition_Target_Inputs>`). The `run <Composition.run>` and `execute <Composition.execute>`
methods can also be used to execute a Composition that has been `configured for learning <Composition_Learning>`,
but no learning will occur, irrespective of the value of the `disable_learning <Composition.disable_learning>`
attribute.

The sections that follow describe the formats that can be used for inputs, factors that impact execution, and
how the results of execution are recorded and reported.


.. _Composition_Execution_Inputs:

*Composition Inputs*
~~~~~~~~~~~~~~~~~~~~

- `Composition_Input_Dictionary`
- `Composition_Programmatic_Inputs`

All `methods of executing <Composition_Execution_Methods>` a Composition require specification of an **inputs**
argument (and a **targets** argument for the `learn <Composition.learn>` method), which designates the values assigned
to the `INPUT <NodeRole.INPUT>` (and, for learning, the `TARGET <NodeRole.TARGET>`) `Nodes <Composition_Nodes>`
of the Composition.  These are provided to the Composition each time it is executed; that is, for each `TRIAL
<TimeScale.TRIAL>`. A `TRIAL <TimeScale.TRIAL>` is defined as the opportunity for every Node in the Composition
to execute the current set of inputs. The inputs for each `TRIAL <TimeScale.TRIAL>` can be specified using an `input
dictionary <Composition_Input_Dictionary>`; for the `run <Composition.run>` and `learn <Composition.learn>` methods,
they can also be specified `programmatically <Composition_Programmatic_Inputs>`. Irrespective of format, the same
number of inputs must be specified for every `INPUT` Node, unless only one value is specified for a Node (in which
case that value is repeated as the input to that Node for every `TRIAL <TimeScale.TRIAL>`\\s executed). If the
**inputs** argument is not specified for the `run <Composition.run>` or `execute <Composition.execute>` methods, the
`default_variable <Component_Variable>` for each `INPUT` Node is used as its input on `TRIAL <TimeScale.TRIAL>`.
If it is not specified for the `learn <Composition.learn>` method, an error is generated unless its **targets**
argument is specified (see `below <Composition_Execution_Learning_Inputs>`).  The Composition's `get_input_format()
<Composition.get_input_format>` method can be used to show an example for how inputs should be formatted for the
Composition, as well as the `INPUT <NodeRole.INPUT>` Nodes to which they are assigned.  The formats are described in
more detail below.

.. _Composition_Input_Formats:

*Input formats (including targets for learning)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to specify inputs:

  * using `a dictionary <Composition_Input_Dictionary>`, in which the inputs are specified or each `TRIAL
    <TimeScale.TRIAL>` explicitly;

  * `programmtically <Composition_Programmatic_Inputs>`, using a function, generator or generator function
    that constructs the inputs dynamically on a `TRIAL <TimeScale.TRIAL>` by `TRIAL <TimeScale.TRIAL>` basis.

The **inputs** argument of the `run <Composition.run>` and `learn <Composition.learn>` methods (and the **targets**
argument of the `learn <Composition.learn>` method) can be specified in either way; however, only the dictionary
format can be used for the `execute <Composition.execute>` method, since it executes only one `TRIAL
<TimeScale.TRIAL>` at a time, and therefore can only accept inputs for asingle `TRIAL <TimeScale.TRIAL>`.

.. _Composition_Input_External_InputPorts:

*Inputs and input_ports*. All formats must specify the inputs to be assigned, on each `TRIAL <TimeScale.TRIAL>`, to
*all* of the **external InputPorts** of the Composition's `INPUT` `Nodes <Composition_Nodes>`. These are InputPorts
belonging to its `INPUT` `Nodes <Composition_Nodes>` at *all levels of nesting*, that are not designated as
`internal_only <InputPort_Internal_Only>`. They are listed in the Composition's `external_input_ports_of_all_input_nodes
<Composition.external_input_ports_of_all_input_nodes>` attribute, as well as the `external_input_ports
<Mechanism_Base.external_input_ports>` attribute of each `Mechanism` that is an `INPUT <NodeRole.INPUT>`
`Node <Composition_Nodes>` of the Composition or any `nested Composition <Composition_Nested>` within it
The format required can also be seen using the  `get_input_format() <Composition.get_input_format>` method.

.. _Composition_Input_Internal_Only:

.. note::
   Most Mechanisms have only a single `InputPort`, and thus require only a single input to be specified for
   them for each `TRIAL <TimeScale.TRIAL>`. However some Mechanisms have more than one InputPort (for example,
   a `ComparatorMechanism`), in which case inputs can be specified for some or all of them (see `below
   <Composition_Input_Dictionary_InputPort_Entries>`). Conversely, some Mechanisms have InputPorts that are designated
   as `internal_only <InputPort.internal_only>` (for example, the `input_port <Mechanism_Base.input_port>` for a
   `RecurrentTransferMechanism`, if its `has_recurrent_input_port <RecurrentTransferMechanism.has_recurrent_input_port>`
   attribute is True), in which case no input should be specified for those input_ports. Similar considerations
   extend to the `external_input_ports_of_all_input_nodes <Composition.external_input_ports_of_all_input_nodes>` of a
   `nested Composition <Composition_Nested>`, based on the Mechanisms (and/or additionally nested Compositions) that
   comprise its set of `INPUT` `Nodes <Composition_Nodes>`.

The factors above determine the format of each entry in an `inputs dictionary <Composition_Input_Dictionary>`, or the
return value of the function or generator used for `programmatic specification <Composition_Programmatic_Inputs>` of
inputs, as described in detail below (also see `examples <Composition_Examples_Input>`).


.. _Composition_Input_Dictionary:

*Input Dictionary*
==================

.. _Composition_Input_Dictionary_Entries:

The simplest way to specificy inputs (including targets for learning) is using a dict, in which each entry specifies
the inputs to a given `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>`. The key for each entry of the dict is
either an `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` or the `InputPort` of one, and the value is the input
to be provided to it for each `TRIAL <TimeScale.TRIAL>` of execution.  A diciontary can have entries for *either* an
INPUT Node or one or more of its InputPorts, but *not both*.  Entries can be for any `INPUT <NodeRole.INPUT>` Node
(or the Inputport(s) of one) at any level of nesting within the Composition, so long it is nested under INPUT Nodes
at all levels of nesting (that is, an INPUT Node of a nested Composition can only be included if the nested Composition
is a INPUT Node of the Composition to which it belongs). Any INPUT Nodes for which no input is specified (that is, for
which there are no entries in the inputs dictionary) are assigned their `default_external_inputs
<Mechanism_Base.default_external_inputs>` on each `TRIAL <TimeScale.TRIAL>` of execution; similarly, if the dictionary
contains entries for some but not all of the InputPorts of a Node, the remaining InputPorts are assigned their
`default_input <InputPort.default_input>` on each `TRIAL <TimeScale.TRIAL>` of execution.  See below for additional
information concerning `entries for Nodes <Composition_Input_Dictionary_Node_Entries>` and `entries for InputPorts
<Composition_Input_Dictionary_InputPort_Entries>`).

.. _Composition_Input_Dictionary_Input_Values:

*Input values*. The value of each entry is an ndarray or nested list containing the inputs to that Node or InputPort.
For Nodes, the value is a 3d array (or correspondingly nested list), in which the outermost items are 2d arrays
containing the 1d array of input values to each of the Node's InputPorts for a given `TRIAL <TimeScale.TRIAL>`. For
entries specifying InputPorts, the value is a 2d array, containing 1d arrays with the input to the InputPort for each
`TRIAL <TimeScale.TRIAL>`. A given entry can specify either a single `TRIAL <TimeScale.TRIAL>`\\'s worth of input
(i.e., a single item in its outermost dimension), or inputs for every `TRIAL <TimeScale.TRIAL>` to be executed (in
which the i-th item represents the input to the `INPUT <NodeRole.INPUT>` Node, or one of its InputPorts, on `TRIAL
<TimeScale.TRIAL>` i).  All entries that contain more than a single trial's worth of input must contain exactly the
same number of values -- i.e.,inputs for the same number of trials. For entries that contain a single input value,
that value will be provided repeatedly as the input to the specified Component for every `TRIAL <TimeScale.TRIAL>`
when the Composition is executed (as determined by the number of input values specified for other entries and/or the
**num_trials** argument of the Composition's `run <Composition.run>` method (see `number of trials
<Composition_Execution_Num_Trials>` above).

.. _Composition_Input_Dictionary_Node_Entries:

*Node entries*.  The key must be an `INPUT <NodeRole>` `Node <Composition_Nodes>` of the Composition, or the name of
one (i.e., the str in its `name <Component.name>` attribute), and the value must specify the input to *all* of its
InputPorts (other than those designated as `internal_only <InputPort.internal_only>`; see `note
<Composition_Input_Internal_Only>` above) for one or all `TRIAL <TimeScale.TRIAL>`\\s of execution.  The values
for each `TRIAL <TimeScale.TRIAL>` must be compatible with each of the corresponding InputPorts (listed in the
`external_input_ports <Mechanism_Base.external_input_ports>` attribute of a Mechanism, and similarly in the
`external_input_ports_of_all_input_nodes <Composition.external_input_ports_of_all_input_nodes>` attribute of a
Composition). More specifically, the shape of each item in the outer dimension (i.e., the input for each `TRIAL
<TimeScale.TRIAL>`, as described `above <Composition_Input_Dictionary_Input_Values>`) must be compatible with the
shape of the Node's `external_input_shape <Mechanism_Base.external_input_shape>` attribute if it is Mechanism, and
similarly the `external_input_shape <Composition.external_input_shape>` attribute of a Composition). While these are
always 2d arrays, the number and size of the 1d arrays within them (corresponding to each InputPort) may vary; in some
case shorthand notations are allowed, as illustrated in the `examples  <Composition_Examples_Input_Dictionary>` below.

    .. _Composition_Execution_Input_Dict_Fig:

    .. figure:: _static/Composition_input_dict_spec.svg
       :alt: Example input dict specification showing inputs specified for each Node and its InputPorts

       Example input dict specification, in which the first entry is for Mechanism ``a`` with one `InputPort` that takes
       an array of length 2 as its input, and for which two `TRIAL <TimesScale.TRIAL>`\\s worth of input are specified
       (``[1.0, 2.0]`` and ``[3,0, 4.0]``);  the second entry is for Mechanism ``b`` with two InputPorts, one of which
       takes an array of length 1 as its input and the other an array of length 2, and for which two `TRIAL
       <TimesScale.TRIAL>`\\s worth of input are also specified (``[[1.0], [2.0, 3.0]]`` and ``[[4.0], [5.0, 6.0]]``);
       and, finaly, a third entry is for Mechanism ``c`` with only one InputPort that takes an array of length 1 as its
       input, and for which only one input is specified (``[1.0]``), which is therefore provided as the input to
       Mechanism ``c`` on every `TRIAL <TimeScale.TRIAL>`.

.. _Composition_Input_Dictionary_InputPort_Entries:

*InputPort Entries*. The key must be an `external InputPort <Composition_Input_External_InputPorts>` for an
`INPUT <NodeRole>` `Node <Composition_Nodes>` of the Composition, or the `full_name <Port_Base.full_name>` of one,
and the value must specify the input for one or all `TRIAL <TimeScale.TRIAL>`\\s of execution.  Any or all of the
InputPorts for an`INPUT <NodeRole>` `Node <Composition_Nodes>` can be specified, but an inputs dictionary cannot
have specifications for both the Node and any of its InputPorts.  If the name of an InputPort is used as the key, its
the str in its `full_name <Port_Base.full_name>` attribute must be used, to ensure disambiguation from any similarly
named InputPorts of other Nodes.  Specifying InputPorts individually (instead of specifying all of them in a single
entry for a Node) can be if only some InputPorts should receive inputs, or the input for some needs to remain constant
across `TRIAL <TimeScale.TRIAL>`\\s (by providing it with only one input value) while the input to others vary
(i.e., by providing input_values for every `TRIAL <TimeScale.TRIAL>`).  The value of each entry must be a 2d array
or nested list containing the input for either a single `TRIAL <TimeScale.TRIAL>` or all `TRIAL <TimeScale.TRIAL>`\\s,
each of which must match the `input_shape <InputPort.input_shape>` of the InputPort. As with Nodes, if there are
entries for some but not all of a Node's InputPorts, the ones not specified are assigned their `default_input
<InputPort.default_input>` values for every `TRIAL <TimeScale.TRIAL>` of execution.

COMMENT:
    Example of input dictionary show various ways of specifying inputs for Nodes and InputPorts::

    >>> A = ComparatorMechanism(name='A')
    >>> B = ProcessingMechanism(name='B', default_variable=[0,0,0])
    >>> inner_nested_comp = Composition(nodes=[A, B])

    >>> C = ComparatorMechanism(name='C', size=3)
    >>> nested_comp_1 = Composition(nodes=[C, inner_nested_comp])

    >>> D = ComparatorMechanism(name='D', size=3)
    >>> E = ComparatorMechanism(name='E', size=3)
    >>> nested_comp_2 = Composition([D, E])

    >>> F = ComparatorMechanism(name='F')
    >>> G = ProcessingMechanism(name='G')
    >>> nested_comp_3 = Composition([F, G])

    >>> H = ComparatorMechanism(name='H')
    >>> I = ProcessingMechanism(name='I')
    >>> outer_comp = Composition(pathways=[nested_comp_1, nested_comp_2], nodes=[H, I, nested_comp_3]])

    >>> input_dict = {A.input_ports['SAMPLE']:[[.5]],  # Note: only a signle TRIAL of input is specified
    ...               A['TARGET']:[[1],[2]],           # Note: name of InputPort is used as key
    ...               B:[[[3,4,5]]],                   # Note: only a signle TRIAL of input is specified
    ...               "C":[[[.5],[1]]],                # Note: name of Node is used as key
    ...               D:[[[6,7,8]],[[9,10,11]]],
    ...               nested_comp_3: [[[12]],[[13]],   # Note: full input nested Composition is provide
    ...                              [[14]],[[15]]]}   #       for each TRIAL of execution
    >>> outer_comp.get_input_format()
    >>> outer_comp.external_input_shape
    >>> outer_comp.external_input_ports_of_all_input_nodes
    >>> outer_comp.run(inputs=inputs)
    Add output here

    Show example of get_input_format()
    Show example of get_external_inputs

    In this example, `ComparatorMechanism` ``A`` has two `InputPorts <InputPort>` that are specified inidividually,
    one of which (``SAMPLE``) has only one `TRIAL <TimeScale.TRIAL>` of input specified, while the other
    (``TARGET``) has two `TRIAL <TimeScale.TRIAL>`\\s of input specified;  ``B`` has only one `TRIAL <TimeScale.TRIAL>`
    specified, the length of which matches the shape of its `default_variable <Component_Variable>`;

    - Add show_graph()
    - A & B are both INPUT Nodes of inner_nested_comp, which is an INPUT node of nested_comp_1, which is an INPUT Node
      of outer_comp, so they count as INPUT Nodes
    - D is an INPUT Node of nested_comp_2, but since nested_comp_2 is *not* an INPUT Node of outer_comp, it is not
      D is not an INPUT Node of outer_comp and thus neiter D nor nested_comp_2 can be listed in the inputs_dict
    - nested_comp_3 is an INPUT Node of outer_comp, so it (as shown in this example) or its INPUT Nodes (F, G) or their
      InputPorts can be included as entries in the inputs dict for outer_comp.
    - inputs for nested_comp_3 must contain all of the InputPorts for all of its InputNodes (F and G): two for F and
      one for G;  all have to have the same number of trials (here two) as each other and all other entries in the
      inputs dictionary.
    - H and I are both INPUT Nodes of outer_comp
    - 'C' is specified by its name
COMMENT

.. _Composition_Input_Labels:

*Input Labels*. In general, the value of inputs should be numeric arrays;  however, some Mechanisms have an
`input_labels_dict
<Mechanism_Base.input_labels_dict>` that specifies a mapping from strings (labels) to numeric values, in which those
strings can be used to specify inputs to that Mechanism (these are translated to their numeric values on execution).
However, such labels are specific to a given Mechanism;  use of strings as input to a Mechanism that does not have an
`input_labels_dict <Mechanism_Base.input_labels_dict>` specified, or use of a string that is not listed in the
dictionary for that Mechanism generates and error.

.. _Composition_Target_Inputs:

*Target Inputs for learning*. Inputs must also be specified for the `TARGET_MECHANISM <Composition_Learning_Components>`
of each `learning Pathway <Composition_Learning_Pathway>` in the Composition. This can be done in either the **inputs**
argument or **targets** argument of the `learn <Composition.learn>` method.  If the **inputs** argument is used,
it must include an entry for each `TARGET_MECHANISM <Composition_Learning_Components>`; if the **targets** argument
is used, it must be assigned a dictionary containing entries in which the key is either an `OUTPUT_MECHANISM
<Composition_Learning_Components>` (i.e., the final `Node <Composition_Nodes>`) of a `learning Pathway
<Composition_Learning_Pathway>`, or the corresponding `TARGET_MECHANISM <Composition_Learning_Components>`. The
value of each entry specifies the inputs for each trial, formatted asdescribed `above <Composition_Input_Dictionary>`.

The input format required for a Composition, and the `INPUT <NodeRole.INPUT>` Nodes to which inputs are assigned,
can be seen using its `get_input_format <Composition.get_input_format>` method.

.. _Composition_Programmatic_Inputs:

*Specifying Inputs Programmatically*
====================================

Inputs can also be specified programmticaly, in a `TRIAL <TimeScale.TRIAL>` by `TRIAL <TimeScale.TRIAL>` manner,
using a function, generator, or generator function.

A function used as input must take as its sole argument the current `TRIAL <TimeScale.TRIAL>` number and return a
value that satisfies all rules above for standard input specification. The only difference is that on each execution,
the function must return the input values for each `INPUT` `Node <Composition_Nodes>` for a single `TRIAL
<TimeScale.TRIAL>`.

.. note::
    Default behavior when passing a function as input to a Composition is to execute for only one `TRIAL
    <TimeScale.TRIAL>`. Remember to set the num_trials argument of Composition.run if you intend to cycle through
    multiple `TRIAL <TimeScale.TRIAL>`\\s.

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
for each `INPUT` `Node <Composition_Nodes>` for a single `TRIAL <TimeScale.TRIAL>`.

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
            # NEW: This function MUST return a dictionary of input values for a single `TRIAL <TimeScale.TRIAL>`
            for each INPUT Node
            return {player: [observation[player_coord_idx]],
                    prey: [observation[prey_coord_idx]]}
        return done
        return {a: [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}

    comp.run(inputs=input_dictionary)
COMMENT


.. _Composition_Execution_Factors:

*Execution Factors*
~~~~~~~~~~~~~~~~~~~

  • `Composition_Runtime_Params`
  • `Composition_Cycles_and_Feedback`
  • `Composition_Execution_Context`
  • `Composition_Timing`
  • `Composition_Reset`
  • `Composition_Compilation`

.. _Composition_Runtime_Params:

*Runtime Parameters*
^^^^^^^^^^^^^^^^^^^^

COMMENT:
    5/8/20
    CHECK THAT runtime_params (AND CONDITIONS) WORK FOR COMPOSITIONS
    ADD EXAMPLES FROM test_runtime_params
    runtime_params are passed to the execute method of the Node whenever it is called for execution
COMMENT

The value of one or more of a Composition's `Nodes <Composition_Nodes>` can be temporarily modified during execution
using the **runtime_params** argument of one of its `execution methods <Composition_Execution_Methods>`.  These are
handled as described for `Mechanism_Runtime_Params` of Mechanisms, with the addition that one or more `Conditions
<Condition>` can be specified such that a value will apply only when the specificied Conditions are satisfied;
otherwise the parameter's previously assigned value (or, if none, then its default) will be used, and those values
are always restored after execution.

.. _Composition_Runtime_Param_Specification:

Runtime parameter values for a Composition are specified in a dictionary assigned to the **runtime_params** argument
of a Composition's `execution method <Composition_Execution_Methods>`.  The key of each entry is a Node of the
Composition, and the value is a subdictionary specifying the **runtime_params** argument that will be passed to the
Node when it is executed.  The format of the dictionary for each `Node <Composition_Nodes>` follows that for a
Mechanism's `runtime specification dictionary <Mechanism_Runtime_Param_Specification>`, except that in addition to
specifying the value of a parameter directly (in which case, the value will apply throughout the execution), its value
can also be placed in a tuple together with a `Condition` specifying when that value should be applied, as follows:

    * Dictionary assigned to **runtime_parms** argument: {<Node>: Runtime Parameter Specification Dictionary}
       - *key* - Node
       - *value* - Runtime Parameter Specification Dictionary

    * Runtime Parameter Specification Dictionary: {<parameter name>: (<parameter value>, `Condition`)}
       - *key* - str
         name of a `Parameter` of the Node, its `function <Mechanism_Base.function>`, or a keyword specifying
         a subdictionary containing runtime parameter specifications for Component(s) of the Node (see below);
       - *value* - (<parameter value>, `Condition`), <parameter value>, or subdictionary (see below) `Condition`
         specifies when the value is applied;  otherwise, its previously assigned value or `default
         <Parameter_Defaults>` is used;  if the parameter values appears alone in a tuple or outside of one,
         then the Condition `Always()` is applied.

    See `Runtime Parameter Specification Dictionary <Mechanism_Runtime_Param_Specification>` for additional details.

As in a standard runtime parameter specification dictionary, the key for an entry can be used to specify a subdictionary
specifying the runtime parameters for a Mechanism's `Ports <Mechanism_Ports>`, and/or any of their `afferent Projections
<Mechanism_Base.afferents>` (see `Mechanism_Runtime_Port_and_Projection_Param_Specification`).  The subdictionaries
used to specify those can be placed placed in a tuple, as can any of the specification of parameter values within them.
A tuple used to specify a subdictionary determines when any of the parameters specified within it are eligible to apply:
If its `Condition` is *not* satisfied, then none of the parameters specified within it will apply;  if its `Condition`
*is* satisfied, then any parameter specified within it for which the `Condition` is satisified will also apply.

.. _Composition_Cycles_and_Feedback:

*Cycles and Feedback*
^^^^^^^^^^^^^^^^^^^^^

    * `Composition_Cycle`
    * `Composition_Feedback`

If `Projections <Projection>` among any or all of the `Nodes <Composition_Nodes>` in a Composition form loops — that
is, there are any cycles in its `graph <Composition_Graph>` —  then the order in which the Nodes are executed must
be determined.  This is handled in one of two ways:

   * **Flatten cycle** - if the cycle does not have any `feedback Projections <Composition_Feedback>`, then the cycle
     is "flattened" and all of the Nodes are executed synchronously.

   * **Break cycle** - if cycle has any `feedback Projections <Composition_Feedback>`, they are used to break the
     cycle at those points, and the remaining Projections are used to execute the Nodes sequentially, with the
     `receiver <Projection_Base.receiver>` of each feedback Projection executed first, its `sender
     <Projection_Base.sender>` executed last, and the receiver getting the sender's value on its next execution.

Each of these approaches is described in greater detail below.

.. _Composition_Cycle:

*Cycles and Synchronous Execution*
==================================


.. _Composition_Cycle_Structure:

**Cycles**. A cycle is formed when the Projections among a set of `Nodes <Composition_Nodes>` form a loop, and none
of the Projections is designated as a `feedback Projection <Composition_Feedback_Designation>`.  Any cycle nested
within another is added to the one in which it is nested, and all are treated as part of the same cycle. All Nodes
within a cycle are assigned the `NodeRole` `CYCLE`.

.. note::
   A `RecurrentTransferMechanism` (and its subclaseses) are treated as single-Node cylces, formed by their
   `AutoAssociativeProjection` (since the latter is subclass of MappingProjection and thus not designated as feedback
   (see `below <Composition_Feedback_Designation>`).

.. _Composition_Cycle_Synchronous_Execution:

**Synchronous execution**. Cycles are "flattened" for execution, meaning that all of the Nodes within a cycle are
executed in the same `TIME_STEP <TimeScale.TIME_STEP>`). The input that each Node in a cycle receives from those that
project to it from within the cycle is the `value <Component.value>` of those Nodes when the cycle was last executed
in the same `execution context <Composition_Execution_Context>`;  this ensures not only that the execute in synchrony,
but that the inputs received from any Nodes within the cycle are synchronized (i.e., from the same earlier `TIME_STEP
<TimeScale.TIME_STEP>` of execution). However, this presents a problem for the first execution of the cycle since, by
definition, none of the Nodes have a value from a previous execution.  In that case, each sender passes the value to
which it has been initialized which, by default, is its `default value <Parameter_Defaults>`.  However, this can be
overridden, as described below.

COMMENT:
 FIGURE HERE
COMMENT

.. note::
   Although all the Nodes in a cycle receive either the initial value or previous value of other Nodes in the cycle,
   they receive the *current* value of any Nodes that project to them from *outisde* the cycle, and pass their current
   value (i.e., the ones computed in the current execution of the cycle) to any Nodes to which they project outside of
   the cycle.  The former means that any Nodes within the cycle that receive such input are "a step ahead" of those
   within the cycle and also, unless the use a `StatefulFunction`, others within the cycle will not see the effects of
   that input within or across `TRIALS <TimeScale.TRIAL>`.

.. _Composition_Cycle_Initialization:

**Initialization**. The initialization of Nodes in a cycle using their `default values <Parameter_Defaults>` can be
overridden using the **initialize_cycle_values** argument of the Composition's `run <Composition.run>` or `learn
<Composition.learn>` methods.  This can be used to specify an initial value for any Node in a cycle.  On the first
call to `run <Composition.run>` or `learn <Composition.learn>`, nodes specified in **initialize_cycle_values** are
initialized using the assigned values, and any Nodes in the cycle that are not specified are assigned their `default
value <Parameter_Defaults>`. In subsequent calls to `run <Composition.run>` or `learn <Composition.learn>`, Nodes
specified in **initialize_cycle_values** will be re-initialized to the assigned values for the first execution of the
cycle in that run, whereas any Nodes not specified will retain the last `value <Component.value>` they were assigned
in the uprevious call to `run <Composition.run>` or `learn <Composition.learn>`.

Nodes in a cycle can also be initialized outside of a call to `run <Composition.run>` or `learn <Composition.run>` using
the `initialize <Composition.initialize>` method.

.. note::
   If a `Mechanism` belonging to a cycle in a Composition is first executed on its own (i.e., using its own `execute
   <Mechanism_Base.execute>` method), the value it is assigned will be used as its initial value when it is executed
   within the Composition, unless an `execution_id <Context.execution_id>` is assigned to the **context** argument
   of the Mechanism's `execute <Mechanism_Base.execute>` method when it is called.  This is because the first time
   a Mechanism is executed in a Composition, its initial value is copied from the `value <Mechanism_Base.value>`
   last assigned in the None context.  As described aove, this can be overridden by specifying an initial value for
   the Mechanism in the **initialize_cycle_values** argument of the call to the Composition's `run <Composition.run>`
   or `learn  <Composition.learn>` methods.

.. _Composition_Feedback:

*Feedback and Sequential Execution*
===================================

.. _Composition_Feedback_Designation:

**Feedback designation**. If any Projections in a loop are `designated as feedback <Composition_Feedback_Designation>`
they are used to break the `cycle <Composition_Cycle_Structure>` of execution that would otherwise be formed, and the
Nodes are executed sequentially as described `below <Composition_Feedback_Sequential_Execution>`.  Each Node that sends
a feedback Projection is assigned the `NodeRole` `FEEDBACK_SENDER`, and the receiver is assigned the `NodeRole`
`FEEDBACK_RECEIVER`.  By default, `MappingProjections <MappingProjection>` are not specified as feedback, and
therefore loops containing only MappingProjections are left as `cycles <Composition_Cycle_Structure>`. In
contrast, `ModulatoryProjections <ModulatoryProjection>` *are* designated as feedback by default, and therefore any
loops containing one or more ModulatoryProjections are broken, with the Mechanism that is `modulated
<ModulatorySignal_Modulation>` designated as the `FEEDBACK_RECEIVER` and the `ModulatoryMechanism` that projects to
it designated as the `FEEDBACK_SENDER`. However, either of these default behaviors can be overidden, by specifying the
feedback status of a Projection explicitly, either in a tuple with the Projection where it is `specified in a Pathway
<Pathway_Specification>` or in the Composition's `add_projections <Composition.add_projections>` method, or by using
the **feedback** argument of the Composition's `add_projection <Composition.add_projection>` method. Specifying True
or the keyword *FEEDBACK* forces its assignment as a *feedback* Projection, whereas False precludes it from being
assigned as a feedback Projection (e.g., a `ControlProjection` that otherwise forms a cycle will no longer do so).

    .. warning::
       Designating a Projection as **feeedback** that is *not* in a loop is allowed, but will issue a warning and
       can produce unexpected results.  Designating more than one Projection as **feedback** within a loop is also
       permitted, by can also lead to complex and unexpected results.  In both cases, the `FEEDBACK_RECEIVER` for any
       Projection designated as **feedback** will receive a value from the Projection that is based either on the
       `FEEDBACK_SENDER`\\'s initial_value (the first time it is executed) or its previous `value <Component.value>`
       (in subsequent executions), rather than its most recently computed `value <Component.value>` whether or not it
       is in a `cycle <Composition_Cycle_Structure>` (see `below <Composition_Feedback_Initialization>`).

.. _Composition_Feedback_Sequential_Execution:

**Sequential execution**. The `FEEDBACK_RECEIVER` is the first of the Nodes that were in a loop to execute in a
given `PASS <TimeScale.PASS>`, receiving a value from the `FEEDBACK_SENDER` as described `below
<Composition_Feedback_Initialization>`.  It is followed in each subsequent `TIME_STEP <TimeScale.TIME_STEP>` by the
next Node in the sequence, with the `FEEDBACK_SENDER` executing last.

.. _Composition_Feedback_Initialization:

**Initialization**. The receiver of a feedback Projection (its `FEEDBACK_RECEIVER`) is treated in the same way as a
`CYCLE` Node:  the first time it executes, it receives input from the `FEEDBACK_SENDER` based on the `value
<Component.value>` to which it was initialized
COMMENT:
   ??CORRECT: ;  however, as with `CYCLE` Node, this can be overidden using....
COMMENT
.  On subsequent executions, its input from the `FEEDBACK_SENDER` is based on the `value <Component.value>` of that
Node after it was last executed in the same `execution context <Composition_Execution_Context>`.

The `FEEDBACK_SENDER`\\s of a Composition are listed in its `feedback_senders <Composition.feedback_senders>` attribute,
and its `FEEDBACK_RECEIVER`\\s  in `feedback_senders <Composition.feedback_senders>`.  These can also be listed using
the Composition's `get_nodes_by_role <Composition.get_nodes_by_role>` method.  The feedback Projections of a Composition
are listed in its `feedback_projections <Composition.feedback_projections>`  attribute, and the feedback status of a
Projection in a Composition is returned by its `get_feedback_status<Composition.get_feedback_status>` method.


.. _Composition_Execution_Context:

*Execution Contexts*
^^^^^^^^^^^^^^^^^^^^

A Composition is always executed in a designated *execution context*, specified by an `execution_id
<Context.execution_id>` that can be provided to the **context** argument of the method used to execute the
Composition. Execution contexts make several capabilities possible, the two most important of which are:

  * a `Component` can be assigned to, and executed in more than one Composition, preserving its `value
    <Component.value>` and that of its `parameters <Parameter_Statefulness>` independently for each of
    the Compositions to which it is assigned;

  * the same Composition can be exectued independently in different contexts; this can be used for
    parallelizing parameter estimation, both for data fitting (see `ParamEstimationFunction`), and for
    simulating the Composition in `model-based optimization <OptimizationControlMechanism_Model_Based>`
    (see `OptimizationControlMechanism`).

If no `execution_id <Context.execution_id>` is specified, the `default execution_id <Composition.default_execution_id>`
is used, which is generally the Composition's `name <Composition.name>`; however, any `hashable
<https://docs.python.org/3/glossary.html>`_ value (e.g., a string, a number, or `Component`) can be used.
That execution_id can then be used to retrieve the `value <Component.value>` of any of the Composition's
Components or their `parameters <Parameter_Statefulness>` that were assigned during the execution. If a Component is
executed outside of a Composition (e.g, a `Mechanism <Mechanism>` is executed on its own using its `execute
<Mechanism_Base.execute>` method), then any assignments to its `value <Component.value>` and/or that of its parameters
is given an execution_id of `None`.

COMMENT:
   MENTION DEFAULT VALUES HERE?  ?= execution_id NONE?
COMMENT

  .. note::
     If the `value <Component.value>` of a Component or a parameter is queried using `dot notation
     <Parameter_Dot_Notation>`, then its most recently assigned value is returned.  To retrieve the
     value associated with a particular execution context, the parameter's `get <Parameter.get>` method must be used:
     ``<Component>.parameters.<parameter_name>.get(execution_id)``, where ``value`` can be used as the parameter_name
     to retrieve the Component's `value <Component.value>`, and the name of any of its other parameters to get their
     value.

See `Composition_Examples_Execution_Context` for examples.

.. technical_note::

    .. _Composition_Execution_Contexts_Init:

    **Initialization of Execution Contexts**

    - The parameter values for any execution context can be copied into another execution context by using
      Component._initialize_from_context, which when called on a Component copies the values for all its parameters
      and recursively for all of the Component's `_dependent_components <Component._dependent_components>`.

    - `_dependent_components <Component._dependent_components>` should be added to for any new Component that requires
      other Components to function properly (beyond "standard" things like Component.function, or Mechanism.input_ports,
      as these are added in the proper classes' _dependent_components).

      - The intent is that with `_dependent_components <Component._dependent_components>` set properly, calling
        ``obj._initialize_from_context(new_context, base_context)`` should be sufficient to run obj under
        **new_context**.

      - A good example of a "nonstandard" override is `OptimizationControlMechanism._dependent_components`

.. _Composition_Timing:

*Timing*
^^^^^^^^

When `run <Composition.run>` is called by a Composition, it calls that Composition's `execute <Composition.execute>`
method once for each `input <Composition_Execution_Inputs>`  (or set of inputs) specified in the call to `run
<Composition.run>`, which constitutes a `TRIAL <TimeScale.TRIAL>` of execution.  For each `TRIAL <TimeScale.TRIAL>`,
the Component makes repeated calls to its `scheduler <Composition.scheduler>`, executing the Components it specifies
in each `TIME_STEP <TimeScale.TIME_STEP>`, until every Component has been executed at least once or another
`termination condition <Scheduler_Termination_Conditions>` is met.  The `scheduler <Composition.scheduler>` can be
used in combination with `Condition` specifications for individual Components to execute different Components at
different time scales.


.. _Composition_Reset:

*Resetting Stateful Parameters*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

COMMENT:
      MOVE TO IntegratorMechanism and relevant arguments of run method
      - DIFERENT FROM INTEGRATOR REINITIALIZATION, WHICH USES/HAS:
        - reset_integrator_nodes_to:  DICT of {node: value} paris
                                   THESE ARE USED AS THE VALUES WHEN EACH NODE'S reset_integrator_when CONDITION IS MET;
                                   IF NOT SPECIIFED, THEN RESET TO previous_value ATTRIBUTE, WHICH IS SET TO THE
                                   MECHANISM'S <object>.defaults.value WHEN CONSTRUCTED
        FIX: ?? CONSIDER MAKING RESET_STATEFUL_FUNCTIONS_WHEN A DICT THAT IS USED TO SUPERCEDE THE attirbute of a Mechanism
        - reset_integrator_nodes_when arg OF RUN: EITHER JUST A Condition, OR A DICT OF {node: Condition} pairs
                                                     IF JUST A CONDITION:  IT SETS THE DEFAULT FOR ALL NODES FOR WHICH
                                                                           THEIR reset_integrator_when ATTRIBUTE IS None
                                                     IF A DICT: IT SETS THE CONDITION FOR ALL SPECIFIFED NODES
                                                                OVERRIDING THEIR SPEC IF THEY HAVE ONE, AND LEAVING ANY
                                                                NOT SPECIFIED IN THE DICT WITH THEIR CURRENT ASSIGNMENT
        - reset_integrator_when attribute of IntegratorMechanism:  SPECIFIES WHEN INTEGRATOR IS RESET UNLESS IT IS NODE
                                                                   INCLUDED IN reset_integrator_nodes_when ARG OF RUN
        - previous_integrator_value attribute (INTERGRATOR) vs. previous value of value attribute (CYCLE)
COMMENT

`Stateful Functions <StatefulFunction>` (such as `IntegratorFunctions <IntegratorFunction>` and "non-parametric"
`MemoryFunctions <MemoryFunction>`) have a `previous_value <StatefulFunction.previous_value>` attribute that maintains
a record of the Function's `values <Parameter.values>` for each `execution context <Composition_Execution_Context>` in
which it is executed, within and between calls to the Composition's `execute methods <Composition_Execution_Methods>`.
This record can be cleared and `previous_value <StatefulFunction.previous_value>` reset to either the Function's
`default value <Parameter_Defaults>` or another one, using either the Composition's `reset <Composition.reset>` method
or in arguments to its `run <Composition.run>` and `learn <Composition.learn>` methods, as described below (see
`Composition_Examples_Reset` for examples):

   * `reset <Composition.reset>` -- this is a method of the Composition that calls the `reset <Component.reset>` method
     of Nodes in the Composition that have a `StatefulFunction`, each of which resets the `stateful parameters
     <Component_Stateful_Parameters>` of those Functions.
     COMMENT:
        ?? OR JUST THEIR `previous_value <StatefulFunction.previous_value>` ??
     COMMENT
     of its `StatefulFunction`. If it is called  without any arguments, it calls the `reset <Component.reset>`
     method for every `Node <Composition_Nodes>` in the Composition that has a `StatefulFunction`.
     It can also be called with a dictionary that specifies a subsset of Nodes to reset (see format descdribed for
     **reset_stateful_functions_when** below).

   * **reset_stateful_functions_when** and **reset_stateful_functions_to** -- these are arguments of the Composition's
     `run <Composition.run>`  and `learn <Composition.learn>` methods, that can be used to specify the `Conditions
     <Condition>` under which the `reset <Component.reset>` method of all or a particular subset of Nodes are called
     during that run, and optionally the values they are assigned when reset.  As with `runtime_params
     <Composition_Runtime_Params>`, these specifications apply only for the duration of the call to `run
     <Composition.run>` or `learn <Composition.learn>`, and the `reset_stateful_function_when
     <Component.reset_stateful_function_when>` of all Nodes are restored to their prior values upon completion.

     - **reset_stateful_functions_when** -- this specifies the `Condition(s) <Condition>` under which the `reset
       <Component.reset>` method will be called for Nodes with `stateful <stateful>`. If a single
       `Condition` is specified, it is applied to all of the Composition's `Nodes <Composition_Nodes>` that have
       `stateful <stateful>`; a dictionary can also be specified, in which the key for each entry
       is a Node, its value is a `Condition` under which that Node's `reset <Component.reset>` method should be called.

       .. note::
           If a single `Condition` is specified, it applies only to Nodes for which the `reset_stateful_function_when
           <Component.reset_stateful_function_when>` attribute has not otherwise been specified. If a dictionary is
           specified, then the `Condition` specified for each Node applies to that Node, superceding any prior
           specification of its `reset_stateful_function_when <Component.reset_stateful_function_when>` attribute
           for the duration of the call to `run <Composition.run>` or `learn <Composition.learn>`.

     - **reset_stateful_functions_to** -- this specifies the values used by each `Node <Composition_Nodes>` to reset
       the `stateful parameters <Component_Stateful_Parameters>` of its `StatefulFunction(s) <StatefulFunction>`.  It
       must be a dictionary of {Node:value} pairs, in which the value specifies the value(s) passed to the
       `reset<Component.reset>` method of the specified Node.  If the `reset <Component.reset>` method of a Node
       takes more than one value (see Note below), then a list of values must be provided (i.e., as {node:[value_0,
       value_1,... value_n]}) that matches the number of arguments taken by the `reset <Component.reset>` method.
       Any Nodes *not* specified in the dictionary are reset using their `default value(s) <Parameter_Defaults>`.

       .. note::
          The `reset <Component.reset>` method of most Nodes with a `StatefulFunction` takes only a single value, that
          is used to reset the `previous_value <StatefulFunction.previous_value>` attribute of the Function.  However
          some (such as those that use `DualAdaptiveIntegrator <DualAdaptiveIntegrator>`) take more than one value.
          For such Nodes, a list of values must be specified as the value of their dicitonary entry in
          **reset_stateful_functions_to**.

     The **reset_stateful_functions_when** and **reset_stateful_functions_to** arguments can be used in conjunction or
     independently of one another. For example, the `Condition(s) <Condition>` under which a `Mechanism` with a
     `StatefulFunction` is reset using to its `default values <Parameter_Defaults>` can be specified by including it in
     **reset_stateful_functions_when** but not **reset_stateful_functions_to**.  Conversely, the value to which
     the `StatefulFunction` of a Mechanism is reset can be specified without changing the Condition under which this
     occurs, by including it in **reset_stateful_functions_to** but not **reset_stateful_functions_when** -- in that
     case, the `Condition` specified by its own `reset_stateful_function_when <Component.reset_stateful_function_when>`
     parameter will be used.


.. _Composition_Compilation:

*Compilation*
^^^^^^^^^^^^^

By default, a Composition is executed using the Python interpreter used to run the script from which it is called. In
many cases, a Composition can also be executed in a `compiled mode <Compilation>`.  While this can add some time to
initiate execution, execution itself can be several orders of magnitude faster than using the Python interpreter. Thus,
using a compiled mode can be useful for executing Compositions that are complex and/or for large numbers of `TRIAL
<TimeScale.TRIAL>`\\s. `Compilation` is supported for most CPUs (including x86, arm64, and powerpc64le).  Several modes
can be specified, that that tradeoff power (i.e., degree of speed-up) against level of support (i.e., likelihood of
success).  Most PsyNeuLink `Components <Component>` and methods are supported for compilation;  however, Python native
functions and methods (e.g., used to specify the `function <Component.function>` of a Component) are not supported at
present. Users who wish to compile custom functions should refer to `compiled User Defined Functions
<UserDefinedFunction>` for more information.   See below and `Compilation` for additional details regarding the use
of compiled modes of execution, and `Vesely et al. (2022) <http://www.cs.yale.edu/homes/abhishek/jvesely-cgo22.pdf>`_
for more information about the approach taken to compilation.

    .. warning::
       Compiled modes are continuing to be developed and refined, and therefore it is still possible that there are
       bugs that will not cause compilation to fail, but could produce erroneous results.  Therefore, it is strongly
       advised that if compilation is used, suitable tests are conducted that the results generated are identical to
       those generated when the Composition is executed using the Python interpreter.

       Users are strongly urged to report any compilation failures to psyneulinkhelp@princeton.edu, or as an
       issue `here <https://github.com/PrincetonUniversity/PsyNeuLink/issues>`_. Known failure conditions are listed
       `here <https://github.com/PrincetonUniversity/PsyNeuLink/milestone/2>`_.

.. _Composition_Compiled_Modes:

The **execution_mode** argument of an `execution method <Composition_Execution_Methods>` specifies whether to use a
compiled mode and, if so,  which.  If True is specified, an attempt is made to use the most powerful mode (LLVMRun)
and, if that fails, to try progressively less powerful modes (issueing a warning indicating the unsupported feature
that caused the failure), reverting to the Python interpreter if all compiled modes fail.  If a particular mode is
specified and fails, an error is generated indicating the unsupported feature that failed. The compiled modes,
in order of their power, are:

.. _Composition_Compilation_LLVM:

    * *True* -- try to use the one that yields the greatesst improvement, progressively reverting to less powerful
      but more forgiving modes, in the order listed below, for each that fails;

    * *LLVMRun* -- compile and run multiple `TRIAL <TimeScale.TRIAL>`\\s; if successful, the compiled binary is
      semantically equivalent to the execution of the `run <Composition.run>` method using the Python interpreter;

    * *LLVMExec* -- compile and run each `TRIAL <TimeScale.TRIAL>`, using the Python interpreter to iterate over them;
      if successful, the compiled binary for each `TRIAL <TimeScale.TRIAL>` is semantically equivalent the execution
      of the `execute <Composition.execute>` method using the Python interpreter;

    * *LLVM* -- compile and run `Node <Composition_Nodes>` of the `Composition` and their `Projections <Projection>`,
      using the Python interpreter to call the Composition's `scheduler <Composition.scheduler>`, execute each Node
      and iterate over `TRIAL <TimeScale.TRIAL>`\\s; note that, in this mode, scheduling `Conditions <Condition>`
      that rely on Node `Parameters` is not supported;

    * *Python* (same as *False*; the default) -- use the Python interpreter to execute the `Composition`.

.. _Composition_Compilation_PTX:

*GPU support.*  In addition to compilation for CPUs, support is being developed for `CUDA
<https://developer.nvidia.com/about-cuda>`_ capable `Invidia GPUs
<https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units>`_.  This can be invoked by specifying one
of the following modes in the **execution_mode** argument of a `Composition execution method
<Composition_Execution_Methods>`:

    * *PTXExec|PTXRun* -- equivalent to the LLVM counterparts but run in a single thread of a CUDA capable GPU.

This requires that a working `pycuda package <https://documen.tician.de/pycuda/>`_ is
`installed <https://wiki.tiker.net/PyCuda/Installation>`_, and that CUDA execution is explicitly enabled by setting
the ``PNL_LLVM_DEBUG`` environment variable to ``cuda``.  At present compilation using these modes runs on a single
GPU thread, and therefore does not produce any performance benefits over running in compiled mode on a CPU;  (see
`this <https://github.com/PrincetonUniversity/PsyNeuLink/projects/1>`_ for progress extending support of parallization
in compiled modes).


.. _Composition_Execution_Results_and_Reporting:

*Results, Reporting and Logging*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _Composition_Execution_Results:

*Results*

Executing a Composition returns the results of the last `TRIAL <TimeScale.TRIAL>` executed. If either `run
<Composition.run>` or `learn <Composition.learn>` is called, the results of all `TRIALS <TimeScale.TRIAL>` executed
are available in the Composition's `results <Composition.results>` attribute.  More specifically, at the end of a
`TRIAL <TimeScale, a Composition's `output_values <Composition.output_values>` (a list of the `output_values
<Mechanism_Base.output_values>` for all of its `OUTPUT <NodeRole.OUTPUT>` `Nodes <Composition_Nodes>`) are added to
the Composition's `results <Composition.results>` attribute, and the `output_values <Mechanism.output_values>` for the
last `TRIAL <TimeScale.TRIAL>` executed is returned by the `execution method <Composition_Execution_Methods>`. The
`output_values <Mechanism_Base.output_values>` of the last `TRIAL <TimeScale.TRIAL>` for each `OUTPUT <NodeRole.OUTPUT>`
can be seen using the Composition's `get_results_by_nodes <Composition.get_results_by_nodes>` method.

.. _Composition_Execution_Reporting:

*Reporting*

A report of the results of each `TRIAL <TimeScale.TRIAL>` can be generated as the Composition is executing, using the
**report_output** and **report_progress** arguments of any of the `execution methods <Composition_Execution_Methods>`.
**report_output** (specified using `ReportOutput` options) generates a report of the input and output of the
Composition and its `Nodes <Composition_Nodes>`, and optionally their `Parameters` (specified in the
**report_params** arg using `ReportParams` options); **report_progress** (specified using `ReportProgress` options)
shows a progress bar  indicating how many `TRIALS <TimeScale.TRIAL>` have been executed and an estimate of the time
remaining to completion.  These options are all OFF by default (see `Report` for additional details).

.. _Composition_Execution_Logging:

*Logging*

The values of individual Components (and their `parameters <Parameters>`) assigned during execution can also be
recorded in their `log <Component_Log>` attribute using the `Log` facility.


.. _Composition_Visualization:

Visualizing a Composition
-------------------------

COMMENT:
    XXX - ADD EXAMPLE OF NESTED COMPOSITION
    XXX - ADD DISCUSSION OF show_controller AND show_learning
COMMENT

The `show_graph <show_graph <ShowGraph.show_graph>` method generates a display of the graph structure of `Nodes
<Composition_Nodes>` and `Projections <Projection>` in the Composition based on the Composition's `graph
<Composition.graph>` (see `Visualization` for additional details).

.. _Composition_Examples:

Composition Examples
--------------------

    * `Composition_Examples_Creation`
    * `Composition_Examples_Run`
    * `Composition_Examples_Learning`
    * `Composition_Examples_Input`
    * `Composition_Examples_Runtime_Params`
    * `Composition_Examples_Cycles_Feedback`
    * `Composition_Examples_Execution_Context`
    * `Composition_Examples_Reset`
    * `Composition_Examples_Visualization`

.. _Composition_Examples_Creation:

*Creating a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~

*Create Mechanisms:*

    >>> import psyneulink as pnl
    >>> A = pnl.ProcessingMechanism(name='A')
    >>> B = pnl.ProcessingMechanism(name='B')
    >>> C = pnl.ProcessingMechanism(name='C')

*Create Projections:*

    >>> A_to_B = pnl.MappingProjection(name="A-to-B")
    >>> B_to_C = pnl.MappingProjection(name="B-to-C")

*Create Composition; Add Nodes (Mechanisms) and Projections using the add_linear_processing_pathway method:*

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

*Nest inner Composition within outer Composition using* `add_node <Composition.add_node>`:

    >>> outer_comp.add_node(inner_comp)

*Create Projections:*

    >>> outer_comp.add_projection(pnl.MappingProjection(), sender=outer_A, receiver=inner_comp)
    >>> outer_comp.add_projection(pnl.MappingProjection(), sender=inner_comp, receiver=outer_B)
    >>> input_dict = {outer_A: [[[1.0]]]}


.. _Composition_Examples_Run:

*Run Composition*
~~~~~~~~~~~~~~~~~

    >>> outer_comp.run(inputs=input_dict)

*Using* `add_linear_processing_pathway <Composition.add_linear_processing_pathway>` *with nested compositions for
brevity:*

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

.. _Composition_Examples_Learning:

*Learning*
~~~~~~~~~~

.. _Composition_Examples_Learning_XOR:

The following example implements a simple three-layered network that learns the XOR function
(see `figure <Composition_Learning_Output_vs_Terminal_Figure>`)::

    # Construct Composition:
    >>> input = TransferMechanism(name='Input', default_variable=np.zeros(2))
    >>> hidden = TransferMechanism(name='Hidden', default_variable=np.zeros(10), function=Logistic())
    >>> output = TransferMechanism(name='Output', default_variable=np.zeros(1), function=Logistic())
    >>> input_weights = MappingProjection(name='Input Weights', matrix=np.random.rand(2,10))
    >>> output_weights = MappingProjection(name='Output Weights', matrix=np.random.rand(10,1))
    >>> xor_comp = Composition('XOR Composition')
    >>> backprop_pathway = xor_comp.add_backpropagation_learning_pathway(
    >>>                       pathway=[input, input_weights, hidden, output_weights, output])

    # Create inputs:            Trial 1  Trial 2  Trial 3  Trial 4
    >>> xor_inputs = {'stimuli':[[0, 0],  [0, 1],  [1, 0],  [1, 1]],
    >>>               'targets':[  [0],     [1],     [1],     [0] ]}
    >>> xor_comp.learn(inputs={input:xor_inputs['stimuli'],
    >>>                      backprop_pathway.target:xor_inputs['targets']},
    >>>              num_trials=1,
    >>>              animate={'show_learning':True})


.. _Composition_Examples_Input:

*Input Formats*
~~~~~~~~~~~~~~~
- `Composition_Examples_Input_Dictionary`
- `Composition_Examples_Programmatic_Input`

.. _Composition_Examples_Input_Dictionary:

Examples of Input Dictionary Specifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is an example in which the **inputs** argument of the `run <Composition.run>` method is specified
as an `input dictionary <Composition_Input_Dictionary>`, with entries for the two `INPUT` `Nodes <Composition_Nodes>`
of the `Composition`::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a',
        ...                           default_variable=[[0.0, 0.0]])
        >>> b = pnl.TransferMechanism(name='b',
        ...                           default_variable=[[0.0], [0.0]])
        >>> c = pnl.TransferMechanism(name='c')

        >>> pathway1 = [a, c]
        >>> pathway2 = [b, c]

        >>> comp = Composition(name='comp', pathways=[patway1, pathway2])

        >>> input_dictionary = {a: [[[1.0, 1.0]], [[1.0, 1.0]]],
        ...                     b: [[[2.0], [3.0]], [[2.0], [3.0]]]}

        >>> comp.run(inputs=input_dictionary)

Since the specification of the `default_variable <Component_Variable>` for Mechanism ``a`` is a single array
of length 2, it is constructed with a single `InputPort` (see `Mechanism_InputPorts`) that takes an array of that
shape as its input; therefore, the input value specified for each `TRIAL <TimeScale.TRIAL>` is a length 2 array
(``[1.0, 1.0]``).  In contrast, since the `default_variable <Component_Variable>` for Mechanism ``b`` is two
length 1 arrays, so it is constructed with two InputPorts, each of which takes a length 1 array as its input;
therefore, the input specified for each `TRIAL <TimeScale.TRIAL>` must be two length 1 arrays.  See `figure
<Composition_Execution_Input_Dict_Fig>` for an illustration of the format for an input dictionary.

COMMENT: MODIFIED 2/4/22 OLD:
# FIX: 2/4/22 - ADD NOTE THAT external_input_values IS NOT NECESSARILY SAME AS external_input_variables
                AS SOME InputPorts CAN HAVE FUNCTIONS THAT CHANGE THE SHAPE OF variable->value (e.g., Reduce)
 # Furthermore, Mechanisms can also have InputPorts with a `function <InputPort.function>` that changes
 #    the size of its input when generatings its `value <InputPort.value>`, in which case its `e
.. note::
    A `Node's <Composition_Nodes>` `external_input_values` attribute is always a 2d list in which the index i
    element is the variable of the i'th element of the Node's `external_input_ports` attribute.  For Mechanisms,
    the `external_input_values <Mechanism_Base.external_input_values>` is often the same as its `variable
    <Mechanism_Base.variable>`.  However, some Mechanisms may have InputPorts marked as `internal_only
    <InputPort.internal_only>` which are excluded from its `external_input_ports <Mechanism_Base.external_input_ports>`
    and therefore its `external_input_values <Mechanism_Base.external_input_values>`, and so should not receive an
    input value.  The same considerations extend to the `external_input_ports <Composition.external_input_ports>`
    and `external_input_values <Composition.external_input_values>` of a Composition, based on the Mechanisms and/or
    `nested Compositions <Composition_Nested>` that comprise its `INPUT` Nodes.
MODIFIED 2/4/22 NEW:
COMMENT
.. note::
    A `Node's <Composition_Nodes>` `external_input_variables` attribute is always a 2d list in which the index i
    element is the variable of the i'th element of the Node's `external_input_ports` attribute.  For Mechanisms,
    the `external_input_variables <Mechanism_Base.external_input_variables>` is often the same as its `variable
    <Mechanism_Base.variable>`.  However, some Mechanisms may have InputPorts marked as `internal_only
    <InputPort.internal_only>` which are excluded from its `external_input_ports <Mechanism_Base.external_input_ports>`
    and therefore its `external_input_variables <Mechanism_Base.external_input_variables>`, and so should not receive
    an input value.  The same considerations extend to the `external_input_ports_of_all_input_nodes
    <Composition.external_input_ports_of_all_input_nodes>` and `external_input_variables
    <Composition.external_input_variables>` of a Composition, based on the Mechanisms and/or `nested Compositions
    <Composition_Nested>` that comprise its `INPUT` Nodes.

If num_trials is not in use, the number of inputs provided determines the number of `TRIAL <TimeScale.TRIAL>`\\s in
the run. For example, if five inputs are provided for each `INPUT` `Node <Composition_Nodes>`, and num_trials is not
specified, the Composition executes five times.

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
are provided for each `INPUT` `Node <Composition_Nodes>`, and num_trials is not specified, the Composition executes
five times., and num_trials = 7, the Composition executes seven times. The input values from `TRIAL
<TimeScale.TRIAL>`\\s 0 and 1 are used again on `TRIAL <TimeScale.TRIAL>`\\s 5 and 6, respectively.

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



For convenience, condensed versions of the input specification described above are also accepted in the following
situations:

* **Case 1:** `INPUT` `Node <Composition_Nodes>` **has only one InputPort**
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

* **Case 2: Only one input is provided for the** `INPUT` `Node <Composition_Nodes>`

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

Shorthand - drop the outer list on **Mechanism a**'s input specification because there is only one
`TRIAL <TimeScale.TRIAL>`:

        >>> input_dictionary = {a: [[1.0], [2.0]]}

        >>> comp.run(inputs=input_dictionary)

* **Case 3: The same input is used on all** `TRIAL <TimeScale.TRIAL>`\\s

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

* **Case 4: There is only one** `INPUT` `Node <Composition_Nodes>`

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

Shorthand - specify **Mechanism a**'s inputs in a list because it is the only `INPUT` `Node <Composition_Nodes>`::

        >>> input_list = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        >>> comp.run(inputs=input_list)
..

.. _Composition_Examples_Programmatic_Input:

Examples of Programmatic Input Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[EXAMPLES TO BE ADDED]

COMMENT:
    EXAMPLES HERE
COMMENT

.. _Composition_Examples_Cycles_Feedback:

*Cycles and Feedback*
~~~~~~~~~~~~~~~~~~~~~

[EXAMPLES TO BE ADDED]
COMMENT:
   EXAMPLES:  [INCLUDE FIGURE WITH show_graph() FOR EACH
        T1 = TransferMechanism(name='T1')
        T2 = TransferMechanism(name='T2')
        T3 = TransferMechanism(name='T3')
        T4 = TransferMechanism(name='T4')
        P4 = MappingProjection(sender=T4, receiver=T1)
        P3 = MappingProjection(sender=T3, receiver=T1)

        # # No feedback version (cycle)
        # C = Composition([T1, T2, T3, T4, P4, T1])
        # C.add_projection(P3)
        C.run({T1:3})
        print('T1: ',T1.value)
        print('T2: ',T2.value)
        print('T3: ',T3.value)
        print('T4: ',T4.value)
        T1:  [[3.]]
        T2:  [[0.]]
        T3:  [[0.]]
        T4:  [[0.]]

        # # One feedback version:
        # C = Composition([T1, T2, T3, T4, (P4, FEEDBACK), T1])
        # C.add_projection(P3)
        C.run({T1:3})
        print('T1: ',T1.value)
        print('T2: ',T2.value)
        print('T3: ',T3.value)
        print('T4: ',T4.value)
        T1:  [[3.]]
        T2:  [[0.]]
        T3:  [[0.]]
        T4:  [[0.]]

        # The other feedback version:
        C = Composition([T1, T2, T3, T4, P4, T1])
        C.add_projection(P3, feedback=FEEDBACK)
        C.run({T1:3})
        print('T1: ',T1.value)
        print('T2: ',T2.value)
        print('T3: ',T3.value)
        print('T4: ',T4.value)
        T1:  [[3.]]
        T2:  [[0.]]
        T3:  [[0.]]
        T4:  [[0.]]

        # # Dual feedback version:
        # C = Composition([T1, T2, T3, T4, (P4, FEEDBACK), T1])
        # C.add_projection(P3, feedback=FEEDBACK)
        C.run({T1:3})
        print('T1: ',T1.value)
        print('T2: ',T2.value)
        print('T3: ',T3.value)
        print('T4: ',T4.value)
        T1:  [[3.]]
        T2:  [[3.]]
        T3:  [[3.]]
        T4:  [[3.]]
COMMENT

.. _Composition_Examples_Runtime_Params:

*Runtime Parameters*
~~~~~~~~~~~~~~~~~~~~

If a runtime parameter is meant to be used throughout the `Run`, then the `Condition` may be omitted and the `Always()`
`Condition` will be assigned by default:

        >>> import psyneulink as pnl

        >>> T = pnl.TransferMechanism()
        >>> C = pnl.Composition(pathways=[T])
        >>> T.function.slope  # slope starts out at 1.0
        1.0

        >>> # During the following run, 10.0 will be used as the slope
        >>> C.run(inputs={T: 2.0},
        ...       runtime_params={T: {"slope": 10.0}})
        [array([20.])]

        >>> T.function.slope  # After the run, T.slope resets to 1.0
        1.0

Otherwise, the runtime parameter value will be used on all executions of the
`Run` during which the `Condition` is True:

        >>> T = pnl.TransferMechanism()
        >>> C = pnl.Composition(pathways=[T])

        >>> T.function.intercept     # intercept starts out at 0.0
        0.0
        >>> T.function.slope         # slope starts out at 1.0
        1.0

        >>> C.run(inputs={T: 2.0},
        ...       runtime_params={T: {"intercept": (5.0, pnl.AfterTrial(1)),
        ...                           "slope": (2.0, pnl.AtTrial(3))}},
        ...       num_trials=5)
        [array([7.])]
        >>> C.results
        [[array([2.])], [array([2.])], [array([7.])], [array([9.])], [array([7.])]]


The table below shows how runtime parameters were applied to the intercept and slope parameters of Mechanism T in the
example above.

+-------------+--------+--------+--------+--------+--------+
|             |Trial 0 |Trial 1 |Trial 2 |Trial 3 |Trial 4 |
+=============+========+========+========+========+========+
| Intercept   |0.0     |0.0     |5.0     |5.0     |5.0     |
+-------------+--------+--------+--------+--------+--------+
| Slope       |1.0     |1.0     |1.0     |2.0     |0.0     |
+-------------+--------+--------+--------+--------+--------+
| Value       |2.0     |2.0     |7.0     |9.0     |7.0     |
+-------------+--------+--------+--------+--------+--------+

as indicated by the results of S.run(), the original parameter values were used on trials 0 and 1,
the runtime intercept was used on trials 2, 3, and 4, and the runtime slope was used on trial 3.

.. note::
    Runtime parameter values are subject to the same type, value, and shape requirements as the original parameter
    value.


.. _Composition_Examples_Execution_Context:

*Execution Contexts*
~~~~~~~~~~~~~~~~~~~~

COMMENT:
  REDUCE REDUNDANCY WITH SECTION ON EXECUTION CONTEXTS ABOVE
COMMENT
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


.. _Composition_Examples_Reset:

*Reset Paramters of Stateful Functions*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example composition with two Mechanisms containing stateful functions:

    >>> A = TransferMechanism(
    >>>     name='A',
    >>>     integrator_mode=True,
    >>>     integration_rate=0.5
    >>> )
    >>> B = TransferMechanism(
    >>>     name='B',
    >>>     integrator_mode=True,
    >>>     integration_rate=0.5
    >>> )
    >>> C = TransferMechanism(name='C')
    >>>
    >>> comp = Composition(
    >>>     pathways=[[A, C], [B, C]]
    >>> )

Example 1) A blanket reset of all stateful functions to their default values:

    >>> comp.run(
    >>>     inputs={A: [1.0],
    >>>             B: [1.0]},
    >>>     num_trials=3
    >>> )
    >>>
    >>> # Trial 1: 0.5, Trial 2: 0.75, Trial 3: 0.875
    >>> print(A.value)
    >>> # [[0.875]]
    >>>
    >>> comp.reset()
    >>>
    >>> # The Mechanisms' stateful functions are now in their default states, which is identical
    >>> # to their states prior to Trial 1. Thus, if we call run again with the same inputs,
    >>> # we see the same results that we saw on Trial 1.
    >>>
    >>> comp.run(
    >>>     inputs={A: [1.0],
    >>>             B: [1.0]},
    >>> )
    >>>
    >>> # Trial 3: 0.5
    >>> print(A.value)
    >>> # [[0.5]]

Example 2) A blanket reset of all stateful functions to custom values:

    >>> comp.run(
    >>>     inputs={A: [1.0],
    >>>             B: [1.0]},
    >>>     num_trials=3
    >>> )
    >>>
    >>> # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.875
    >>> print(A.value)
    >>> # [[0.875]]
    >>> # Mechanism B is currently identical to Mechanism A
    >>> print(B.value)
    >>> # [[0.875]]
    >>>
    >>> # Reset Mechanism A to 0.5 and Mechanism B to 0.75, corresponding to their values at the end of
    >>> # trials 0 and 1, respectively. Thus, if we call run again with the same inputs, the values of the
    >>> # Mechanisms will match their values after trials 1 and 2, respectively.
    >>> comp.reset({A: 0.5,
    >>>             B: 0.75})
    >>>
    >>> comp.run(
    >>>     inputs={A: [1.0],
    >>>             B: [1.0]},
    >>> )
    >>>
    >>> # Trial 3: 0.75
    >>> print(A.value)
    >>> # [[0.75]]
    >>>
    >>> # Trial 3: 0.875
    >>> print(B.value)
    >>> # [[0.875]]

Example 3) Schedule resets for both Mechanisms to their default values, to occur at different times

    >>> comp.run(
    >>>     inputs={A: [1.0],
    >>>             B: [1.0]},
    >>>     reset_stateful_functions_when = {
    >>>         A: AtTrial(1),
    >>>         B: AtTrial(2)
    >>>     },
    >>>     num_trials=5
    >>> )
    >>> # Mechanism A - resets to its default (0) at the beginning of Trial 1. Its value at the end of Trial 1 will
    >>> # be exactly one step of integration forward from its default.
    >>> # Trial 0: 0.5, Trial 1: 0.5, Trial 2: 0.75, Trial 3: 0.875, Trial 4: 0.9375
    >>> print(A.value)
    >>> # [[0.9375]]
    >>>
    >>> # Mechanism B - resets to its default (0) at the beginning of Trial 2. Its value at the end of Trial 2 will
    >>> # be exactly one step of integration forward from its default.
    >>> # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
    >>> print(B.value)
    >>> # [[0.875]]

Example 4) Schedule resets for both Mechanisms to custom values, to occur at different times

    >>> comp.run(
    >>>     inputs={A: [1.0],
    >>>             B: [1.0]},
    >>>     reset_stateful_functions_when={
    >>>         A: AtTrial(3),
    >>>         B: AtTrial(4)
    >>>     },
    >>>     reset_stateful_functions_to={
    >>>         A: 0.5,
    >>>         B: 0.5
    >>>     },
    >>>     num_trials=5
    >>> )
    >>> # Mechanism A - resets to 0.5 at the beginning of Trial 3. Its value at the end of Trial 3 will
    >>> # be exactly one step of integration forward from 0.5.
    >>> # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.875, Trial 3: 0.75, Trial 4:  0.875
    >>> print(A.value)
    >>> # [[0.875]]
    >>>
    >>> # Mechanism B - resets to 0.5 at the beginning of Trial 4. Its value at the end of Trial 4 will
    >>> # be exactly one step of integration forward from 0.5.
    >>> # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.875, Trial 3: 0.9375. Trial 4: 0.75
    >>> print(B.value)
    >>> # [[0.75]]

.. _Composition_Class_Reference:

Class Reference
---------------

"""

import collections
import enum
import functools
import inspect
import itertools
import logging
import sys
import typing
import warnings
from copy import deepcopy, copy
from inspect import isgenerator, isgeneratorfunction
from typing import Union

import graph_scheduler
import networkx
import numpy as np
import pint
import typecheck as tc
from PIL import Image

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import Component, ComponentsMeta
from psyneulink.core.components.functions.fitfunctions import make_likelihood_function
from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.nonstateful.combinationfunctions import LinearCombination, \
    PredictionErrorDeltaFunction
from psyneulink.core.components.functions.nonstateful.learningfunctions import \
    LearningFunction, Reinforcement, BackPropagation, TDLearning
from psyneulink.core.components.functions.nonstateful.transferfunctions import Identity
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base, MechanismError, MechanismList
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import AGENT_REP, \
    OptimizationControlMechanism
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    LearningMechanism, ACTIVATION_INPUT_INDEX, ACTIVATION_OUTPUT_INDEX, ERROR_SIGNAL, ERROR_SIGNAL_INDEX
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.ports.inputport import InputPort, InputPortError
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.port import Port, PortError
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection, MappingError
from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base
from psyneulink.core.components.projections.projection import \
    Projection_Base, ProjectionError, DuplicateProjectionError
from psyneulink.core.components.shellclasses import Composition_Base
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.compositions.report import Report, \
    ReportOutput, ReportParams, ReportProgress, ReportSimulations, ReportDevices, \
    EXECUTE_REPORT, CONTROLLER_REPORT, RUN_REPORT, PROGRESS_REPORT
from psyneulink.core.compositions.showgraph import ShowGraph, INITIAL_FRAME, SHOW_CIM, EXECUTION_SET, SHOW_CONTROLLER
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    AFTER, ALL, ALLOW_PROBES, ANY, BEFORE, COMPONENT, COMPOSITION, CONTROL, CONTROL_SIGNAL, CONTROLLER, DEFAULT, \
    DICT, FEEDBACK, FULL, FUNCTION, HARD_CLAMP, IDENTITY_MATRIX, INPUT, INPUT_PORTS, INPUTS, INPUT_CIM_NAME, \
    LEARNED_PROJECTIONS, LEARNING_FUNCTION, LEARNING_MECHANISM, LEARNING_MECHANISMS, LEARNING_PATHWAY, \
    MATRIX, MATRIX_KEYWORD_VALUES, MAYBE, \
    MODEL_SPEC_ID_METADATA, \
    MONITOR, MONITOR_FOR_CONTROL, NAME, NESTED, NO_CLAMP, NODE, OBJECTIVE_MECHANISM, ONLINE, OUTCOME, \
    OUTPUT, OUTPUT_CIM_NAME, OUTPUT_MECHANISM, OUTPUT_PORTS, OWNER_VALUE, \
    PARAMETER, PARAMETER_CIM_NAME, PORT, \
    PROCESSING_PATHWAY, PROJECTION, PROJECTION_TYPE, PROJECTION_PARAMS, PULSE_CLAMP, RECEIVER, \
    SAMPLE, SENDER, SHADOW_INPUTS, SOFT_CLAMP, SSE, \
    TARGET, TARGET_MECHANISM, TEXT, VARIABLE, WEIGHT, OWNER_MECH
from psyneulink.core.globals.log import CompositionLog, LogCondition
from psyneulink.core.globals.parameters import Parameter, ParametersBase, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel, _assign_prefs
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import ContentAddressableList, call_with_pruned_args, convert_to_list, \
    nesting_depth, convert_to_np_array, is_numeric, is_matrix, parse_valid_identifier
from psyneulink.core.scheduling.condition import All, AllHaveRun, Always, Any, Condition, Never
from psyneulink.core.scheduling.scheduler import Scheduler, SchedulingMode
from psyneulink.core.scheduling.time import Time, TimeScale
from psyneulink.library.components.mechanisms.modulatory.learning.autoassociativelearningmechanism import \
    AutoAssociativeLearningMechanism
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism, MSE
from psyneulink.library.components.mechanisms.processing.objective.predictionerrormechanism import \
    PredictionErrorMechanism
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import \
    RecurrentTransferMechanism
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

__all__ = [
    'Composition', 'CompositionError', 'CompositionRegistry', 'EdgeType', 'get_compositions', 'NodeRole'
    ]

logger = logging.getLogger(__name__)

CompositionRegistry = {}


class CompositionError(Exception):

    def __init__(self, error_value, **kwargs):
        self.error_value = error_value
        self.return_items = kwargs

    def __str__(self):
        return repr(self.error_value)


class RunError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EdgeType(enum.Enum):
    """
        Attributes:
            NON_FEEDBACK
                A standard edge that if it exists in a cycle will only be flattened, not pruned

            FEEDBACK
                A "feedbacK" edge that will be immediately pruned to create an acyclic graph

            FLEXIBLE
                An edge that will be pruned only if it exists in a cycle
    """
    NON_FEEDBACK = 0
    FEEDBACK = 1
    FLEXIBLE = 2


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

        # when pruning a vertex for a processing graph, we store the connection type (the vertex.feedback)
        # to the new child or parent here
        # self.source_types = collections.defaultdict(EdgeType.NORMAL)
        self.source_types = {}

    def __repr__(self):
        return '(Vertex {0} {1})'.format(id(self), self.component)

    @property
    def feedback(self):
        return self._feedback

    @feedback.setter
    def feedback(self, value: EdgeType):
        mapping = {
            False: EdgeType.NON_FEEDBACK,
            True: EdgeType.FEEDBACK,
            FEEDBACK: EdgeType.FEEDBACK,
            MAYBE: EdgeType.FLEXIBLE
        }
        try:
            self._feedback = mapping[value]
        except KeyError:
            self._feedback = value


class Graph(object):
    """A Graph of vertices and edges.

    Attributes
    ----------

    comp_to_vertex : Dict[`Component <Component>` : `Vertex`]
        maps `Component` in the graph to the `Vertices <Vertex>` that represent them.

    vertices : List[Vertex]
        the `Vertices <Vertex>` contained in this Graph;  each can be a `Node <Composition_Nodes>` or a
        `Projection <Component_Projections>`.

    dependency_dict : Dict[`Component` : Set(`Component`)]
        maps each of the graph's Components to the others from which it receives input
        (i.e., their `value <Component.value>`).  For a `Node <Components_Nodes>`, this is one or more
        `Projections <Projection>`;  for a Projection, it is a single Node.

    """

    def __init__(self):
        self.comp_to_vertex = collections.OrderedDict()  # Translate from PNL Mech, Comp or Proj to corresponding vertex
        self.vertices = []  # List of vertices within graph

        self.cycle_vertices = set()

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

            list[`Vertex`] :
              list of the parent `Vertices <Vertex>` of the Vertex associated with **component**.
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

            list[`Vertex`] :
                list of the child `Vertices <Vertex>` of the Vertex associated with **component**.
        """
        return self.comp_to_vertex[component].children

    def prune_feedback_edges(self):
        """
            Produces an acyclic graph from this Graph. `Feedback <EdgeType.FEEDBACK>` edges are pruned, as well as
            any edges that are `potentially feedback <EdgeType.FLEXIBLE>` that are in cycles. After these edges are
            removed, if cycles still remain, they are "flattened." That is, each edge in the cycle is pruned, and
            each the dependencies of each Node in the cycle are set to the pre-flattened union of all cyclic nodes'
            parents that are themselves not in a cycle.

            Returns:
                a tuple containing
                - the acyclic dependency dictionary produced from this
                Graph
                - a dependency dictionary containing only the edges
                removed to create the acyclic graph
                - the unmodified cyclic dependency dictionary of this
                Graph
        """

        # stores a modified version of the self in which cycles are "flattened"
        execution_dependencies = self.dependency_dict
        # stores the original unmodified dependencies
        structural_dependencies = self.dependency_dict
        # wipe and reconstruct list of vertices in cycles
        self.cycle_vertices = set()
        flexible_edges = set()

        for node in execution_dependencies:
            # prune recurrent edges
            try:
                execution_dependencies[node].remove(node)
                self.cycle_vertices.add(node)
            except KeyError:
                pass

            for dep in tuple(execution_dependencies[node]):
                vert = self.comp_to_vertex[node]
                dep_vert = self.comp_to_vertex[dep]

                if dep_vert in vert.source_types:
                    # prune standard edges labeled as feedback
                    if vert.source_types[dep_vert] is EdgeType.FEEDBACK:
                        execution_dependencies[node].remove(dep)
                    # store flexible edges for potential pruning later
                    elif vert.source_types[dep_vert] is EdgeType.FLEXIBLE:
                        flexible_edges.add((dep, node))

        # construct a parallel networkx graph to use its cycle algorithms
        nx_graph = self._generate_networkx_graph(execution_dependencies)
        connected_components = list(networkx.strongly_connected_components(nx_graph))

        # prune only one flexible edge per attempt, to remove as few
        # edges as possible
        # For now, just prune the first flexible edge each time. Maybe
        # look for "best" edges to prune in future by frequency in
        # cycles, if that occurs
        for parent, child in flexible_edges:
            cycles = [c for c in connected_components if len(c) > 1]

            if len(cycles) == 0:
                break

            if any((parent in c and child in c) for c in cycles):
                # prune
                execution_dependencies[child].remove(parent)
                self.comp_to_vertex[child].source_types[self.comp_to_vertex[parent]] = EdgeType.FEEDBACK
                nx_graph.remove_edge(parent, child)
                # recompute cycles after each prune
                connected_components = list(networkx.strongly_connected_components(nx_graph))

        # find all the parent nodes for each node in a cycle, excluding
        # parents that are part of the cycle
        for cycle in [c for c in connected_components if len(c) > 1]:
            acyclic_dependencies = set()

            for node in cycle:
                acyclic_dependencies = acyclic_dependencies.union({
                    parent for parent in execution_dependencies[node]
                    if parent not in cycle
                })

            # replace the dependencies of each node in the cycle with
            # each of the above parents outside of the cycle. This
            # ensures that they all share the same parents and will then
            # exist in the same consideration set

            # NOTE: it is unnecessary to change any childrens'
            # dependencies because any child dependent on a node n_i in
            # a cycle will still depend on n_i when it is part of a
            # flattened cycle. The flattened cycle will simply add more
            # nodes to the consideration set in which n_i exists
            for child in cycle:
                self.cycle_vertices.add(child)
                execution_dependencies[child] = acyclic_dependencies

        return (
            execution_dependencies,
            {
                node: structural_dependencies[node] - execution_dependencies[node]
                for node in execution_dependencies
            },
            structural_dependencies
        )

    def get_strongly_connected_components(
        self,
        nx_graph: typing.Optional[networkx.DiGraph] = None
    ):
        if nx_graph is None:
            nx_graph = self._generate_networkx_graph()

        return list(networkx.strongly_connected_components(nx_graph))

    def _generate_networkx_graph(self, dependency_dict=None):
        if dependency_dict is None:
            dependency_dict = self.dependency_dict

        nx_graph = networkx.DiGraph()
        nx_graph.add_nodes_from(list(dependency_dict.keys()))
        for child in dependency_dict:
            for parent in dependency_dict[child]:
                nx_graph.add_edge(parent, child)

        return nx_graph

    @property
    def dependency_dict(self):
        return dict((v.component,set(d.component for d in v.parents)) for v in self.vertices)


class NodeRole(enum.Enum):
    """Roles assigned to `Nodes <Composition_Nodes>` of a `Composition`.

    Attributes
    ----------

    ORIGIN
        A `Node <Composition_Nodes>` that does not receive any `Projections <Projection>` from any other Nodes
        within its own `Composition`, though if it is in a `nested Composition <Composition_Nested>` it may
        receive Projections from the outer Composition.  `Execution of a `Composition <Composition_Execution>`
        always begins with an `ORIGIN` Node.  A Composition may have many `ORIGIN` Nodes.  This role cannot be
        modified programmatically.

    INPUT
        A `Node <Composition_Nodes>` that receives input from outside its `Composition`, either from the Composition's
        `run <Compositions.run>` method or, if it is in a `nested Composition <Composition_Nested>`, from the outer
        Composition.  By default, the `ORIGIN` Nodes of a Composition are also its `INPUT` Nodes; however this can be
        modified by `assigning specified NodeRoles <Composition_Node_Role_Assignment>` to Nodes.  A Composition can
        have many `INPUT` Nodes.  Note that any Node that `shadows <InputPort_Shadow_Inputs>` an `INPUT` Node is itself
        also assigned the role of `INPUT` Node.

    PROBE
        A `Node <Composition_Nodes>` that is neither `ORIGIN` nor `TERMINAL` but that is treated as an

    SINGLETON
        A `Node <Composition_Nodes>` that is both an `ORIGIN` and a `TERMINAL`.  This role cannot be modified
        programmatically.

    INTERNAL
        A `Node <Composition_Nodes>` that is neither `INPUT` nor `OUTPUT`.  Note that it *can* also be `ORIGIN`,
        `TERMINAL` or `SINGLETON`, if it has no `afferent <Mechanism_Base.afferents>` or `efferent
        <Mechanism_Base.efferents>` Projections or neither, respectively. This role cannot be modified programmatically.

    CYCLE
        A `Node <Composition_Nodes>` that belongs to a cycle. This role cannot be modified programmatically.

    FEEDBACK_SENDER
        A `Node <Composition_Nodes>` with one or more efferent `Projections <Projection>` designated as `feedback
        <Composition_Feedback_Designation>` in the Composition.  This means that the Node executes last in the
        sequence of Nodes that would otherwise form a `cycle <Composition_Cycle_Structure>`. This role cannot be
        modified directly, but is modified if the feedback status` of the Projection is `explicitly specified
        <Composition_Feedback_Designation>`.

    FEEDBACK_RECEIVER
        A `Node <Composition_Nodes>` with one or more afferent `Projections <Projection>` designated as `feedback
        <Composition_Feedback_Designation>` in the Composition. This means that the Node executes first in the
        sequence of Nodes that would otherwise form a `cycle <Composition_Cycle_Structure>`. This role cannot be
        modified directly, but is modified if the feedback status` of the Projection is `explicitly specified
        <Composition_Feedback_Designation>`.

    CONTROL_OBJECTIVE
        A `Node <Composition_Nodes>` that is an `ObjectiveMechanism` associated with a `ControlMechanism` other
        than the Composition's `controller <Composition.controller>` (if it has one).

    CONTROLLER
        A `Node <Composition_Nodes>` that is the `controller <Composition.controller>` of a Composition.
        This role cannot be modified programmatically.

    CONTROLLER_OBJECTIVE
        A `Node <Composition_Nodes>` that is an `ObjectiveMechanism` associated with a Composition's `controller
        <Composition.controller>`.

    LEARNING
        A `Node <Composition_Nodes>` that is only executed when learning is enabled;  if it is not also assigned
        `TARGET` or `LEARNING_OBJECTIVE`, then it is a `LearningMechanism`. This role can, but generally should not be
        modified programmatically.

    COMMENT:
    LEARNING_OUTPUT
        A `Node <Composition_Nodes>` that is last one in a `learning Pathway <Composition_Learning_Pathway>`,
        the desired `value <Mechanism_Base.value>` of which is provided as input to the `TARGET_MECHANISM
        <Composition_Learning_Components>` for that pathway (see `OUTPUT_MECHANISM
        <Composition_Learning_Components>`. This role can, but generally should not be modified programmatically.
    COMMENT

    TARGET
        A `Node <Composition_Nodes>` that receives the target for a `learning pathway
        <Composition_Learning_Pathway>` specifying the desired output of the `OUTPUT_MECHANISM
        <Composition_Learning_Components>` for that pathway (see `TARGET_MECHANISM <Composition_Learning_Components>`).
        This role can, but generally should not be modified programmatically.

    LEARNING_OBJECTIVE
        A `Node <Composition_Nodes>` that is the `ObjectiveMechanism` of a `learning Pathway
        <Composition_Learning_Pathway>`; usually a `ComparatorMechanism` (see `OBJECTIVE_MECHANISM`). This role can,
        but generally should not be modified programmatically.

    PROBE
        An `INTERNAL` `Node <Composition_Nodes>` that is permitted to have Projections from it to the Composition's
        `output_CIM <Composition.output_CIM>`, but -- unlike an `OUTPUT` Node -- the `output_values
        <Mechanism_Base.output_values>` of which are *not* included in the Composition's `results
        <Composition.results>` attribute (see `allow_probes <OptimizationContorlMechanism.allow_probes>` for an
        example.

    OUTPUT
        A `Node <Composition_Nodes>` the `output_values <Mechanism_Base.output_values>` of which are included in
        the Composition's `results <Composition.results>` attribute.  By default, the `TERMINAL` Nodes of a
        Composition are also its `OUTPUT` Nodes; however this can be modified by `assigning specified NodeRoles
        <Composition_Node_Role_Assignment>` to Nodes.  A Composition can have many `OUTPUT` Nodes.

    TERMINAL
        A `Node <Composition_Nodes>` that does not send any `Projections <Projection>` to any other Nodes within
        its own `Composition`, though if it is in a `nested Composition <Composition_Nested>` it may send Projections
        to the outer Composition. A Composition may have many `TERMINAL` Nodes. The `ObjectiveMechanism` associated
        with the Composition's `controller <Composition.controller>` (assigned the role `CONTROLLER_OBJECTIVE`)
        cannot be a `TERMINAL` Node of a Composition.  `Execution of a Composition <Composition_Execution>` itself
        always ends with a `TERMINAL` Node, although the `controller <Composition.controller>` and its associated
        `ObjectiveMechanism` may execute after that; some `TERMINAL` Nodes may also execute earlier (i.e., if they
        belong to a `Pathway` that is shorter than the longest one in the Composition).
        This role cannot be modified programmatically.

    """
    ORIGIN = enum.auto()
    INPUT = enum.auto()
    SINGLETON = enum.auto()
    INTERNAL = enum.auto()
    CYCLE = enum.auto()
    FEEDBACK_SENDER = enum.auto()
    FEEDBACK_RECEIVER = enum.auto()
    CONTROL_OBJECTIVE = enum.auto()
    CONTROLLER = enum.auto()
    CONTROLLER_OBJECTIVE = enum.auto()
    LEARNING = enum.auto()
    TARGET = enum.auto()
    LEARNING_OBJECTIVE = enum.auto()
    PROBE = enum.auto()
    OUTPUT = enum.auto()
    TERMINAL = enum.auto()


class Composition(Composition_Base, metaclass=ComponentsMeta):
    """
    Composition(                           \
        pathways=None,                     \
        nodes=None,                        \
        projections=None,                  \
        allow_probes=True,                 \
        include_probes_in_output=False     \
        disable_learning=False,            \
        controller=None,                   \
        enable_controller=None,            \
        controller_mode=AFTER,             \
        controller_time_scale=TRIAL        \
        controller_condition=Always(),     \
        retain_old_simulation_data=None,   \
        show_graph_attributes=None,        \
        name=None,                         \
        prefs=Composition.classPreference  \
        )

    Base class for Composition.

    Arguments
    ---------

    pathways : Pathway specification or list[Pathway specification...]
        specifies one or more Pathways to add to the Compositions. A list containing `Node <Composition_Nodes>`
        and possible `Projection` specifications at its top level is treated as a single `Pathway`; a list containing
        any nested lists or other forms of `Pathway specification <Pathway_Specification_Formats>` is treated as
        `multiple pathways <Pathway_Specification_Multiple>` (see `pathways <Composition_Pathways_Arg>` as
        well as `Pathway specification <Pathway_Specification>` for additional details).

        .. technical_note::

           The design pattern for use of sets and lists in specifying the **pathways** argument are:
             - sets comprise Nodes that all occupy the same (parallel) position within a processing Pathway;
             - lists comprise *sequences* of Nodes; embedded list are either ignored or a generate an error (see below)
               (this is because lists of Nodes are interpreted as Pathways and Pathways cannot be nested, which would be
               redundant since the same can be accomplished by simply including the items "inline" within a single list)
             - if the Pathway specification contains (in its outer list):
                 - only a single item or set of items, each is treated as a SINGLETON <NodeRole.SINGLETON> in a Pathway;
                 - one or more lists, the items in each list are treated as separate (parallel) pathways;
                 - singly-nested lists ([[[A,B]],[[C,D]]]}), they are collapsed and treated as a Pathway;
                 - any list with more than one list nested within it ([[[A,B],[C,D]}), an error is generated;
                 - Pathway objects are treated as a list (if its pathway attribute is a set, it is wrapped in a list)
             (see `tests <test_various_pathway_configurations_in_constructor>` for examples)

    nodes : `Mechanism <Mechanism>`, `Composition` or list[`Mechanism <Mechanism>`, `Composition`] : default None
        specifies one or more `Nodes <Composition_Nodes>` to add to the Composition;  these are each treated as
        `SINGLETON <NodeRole.SINGLETON>`\\s unless they are explicitly assigned `Projections <Projection>`.

    projections : `Projection <Projection>` or list[`Projection <Projection>`] : default None
        specifies one or more `Projections <Projection>` to add to the Composition;  these are not functional
        unless they are explicitly assigned a `sender <Projection.sender>` and `receiver <Projection.receiver>`.

    allow_probes : bool : default True
        specifies whether `Projections <Projection>` are allowed from `Nodes <Composition_Nodes>` of a `nested
        Composition <Composition_Nested>` other than its OUTPUT <NodeRole.OUTPUT>` `Nodes <Composition_Nodes>` to
        Nodes in outer Composition(s) (see `allow_probes <Composition.allow_probes>` for additional information).

    include_probes_in_output : bool : default False
        specifies whether the outputs of `PROBE <NodeRole.PROBE>` Nodes within a `nested Composition
        <Composition_Nested>` are included in the `output_values <Composition.output_values>` and `results
        <Composition.results>` of the Composition to which they project If False, the outputs of `PROBE
        <NodeRole.PROBE>` Nodes *are excluded* from those attributes;  if True (the default) they are included
        (see `Probes <Composition_Probes>` for additional details).

    disable_learning: bool : default False
        specifies whether `LearningMechanisms <LearningMechanism>` in the Composition are executed when run in
        `learning mode <Composition.learn>`.

    controller : `OptimizationControlMechanism` : default None
        specifies the `OptimizationControlMechanism` to use as the `Composition's controller
        <Composition_Controller>`.

    enable_controller: bool : default None
        specifies whether the Composition's `controller <Composition.controller>` is executed when the
        Composition is run.  Set to True by default if **controller** specified (see `enable_controller
        <Composition.enable_controller>` for additional details).

    controller_mode: enum.Enum[BEFORE|AFTER] : default AFTER
        specifies whether the `controller <Composition.controller>` is executed before or after the rest of the
        Composition when it is run, at the `TimeScale` specified by **controller_time_scale**). Must be either the
        keyword *BEFORE* or *AFTER* (see `controller_mode <Composition.controller_mode>` for additional details).

    controller_time_scale: TimeScale[TIME_STEP, PASS, TRIAL, RUN] : default TRIAL
        specifies the frequency at which the `controller <Composition.controller>` is executed, either before or
        after the Composition is run as specified by **controller_mode** (see `controller_time_scale
        <Composition.controller_time_scale>` for additional details).

    controller_condition: Condition : default Always()
        specifies a specific `Condition` for whether the Composition's `controller <Composition.controller>` is
        executed  in a trial (see `controller_condition <Composition.controller_condition>` for additional details).

    retain_old_simulation_data : bool : default False
        specifies whether or not to retain Parameter values generated during `simulations
        <OptimizationControlMechanism_Execution>` of the Composition (see `retain_old_simulation_data
        <Composition.retain_old_simulation_data>` for additional details).

    show_graph_attributes : dict : None
        specifies features of how the Composition is displayed when its `show_graph <ShowGraph.show_graph>`
        method is called or **animate** is specified in a call to its `run <Composition.run>` method
        (see `ShowGraph` for list of attributes and their values).

    name : str : default see `name <Composition.name>`
        specifies the name of the Composition.

    prefs : PreferenceSet or specification dict : default Composition.classPreferences
        specifies the `PreferenceSet` for the Composition; see `prefs <Composition.prefs>` for details.

    Attributes
    ----------

    graph : `Graph`
        the full `Graph` associated with the Composition. Contains both `Nodes <Composition_Nodes>`
        (`Mechanisms <Mechanism>` or `Compositions <Composition>`) and `Projections <Projection>`.

    nodes : ContentAddressableList[`Mechanism <Mechanism>` or `Composition`]
        a list of all `Nodes <Composition_Nodes>` in the Composition.

    node_ordering : list[`Mechanism <Mechanism>` or `Composition`]
        a list of all `Nodes <Composition_Nodes>` in the order in which they were added to the Composition.
        COMMENT:
            FIX: HOW IS THIS DIFFERENT THAN Composition.nodes?
        COMMENT

    allow_probes : bool or CONTROL
        indicates whether `Projections <Projection>` are allowed to `Nodes <Composition_Nodes>` in the Composition
        from ones of a `nested Composition <Composition_Nested>` other than its OUTPUT <NodeRole.OUTPUT>` `Nodes
        <Composition_Nodes>`.  If *allow_probes* is False, Projections can be received from only the `OUTPUT
        <NodeRole.OUTPUT>` Nodes of a nested Composition;  if it is True (the default), Projections can be received
        from any Nodes of a nested Composition, including its `INPUT <NodeRole.INPUT>` and `INTERNAL
        <NodeRole.INTERNAL>` Nodes;  if it is assigned *CONTROL*, then only the Composition's `controller
        <Composition.controller>` or its `objective_mechanism <ControlMechanism.objetive_mechanism>` can receive
        Projections from such Nodes.  Any Nodes of a nested Composition that project to an enclosing Composition,
        other than its `OUTPUT <NodeRole.OUTPUT>` Nodes, are assigned `PROBE <NodeRole.PROBE>` in addition to their
        other roles (see `Probes <Composition_Probes>` for additional information).

    include_probes_in_output : bool : default False
        determines whether the outputs of `PROBE <NodeRole.PROBE>` Nodes within a `nested Composition
        <Composition_Nested>` are included in the `output_values <Composition.output_values>` and `results
        <Composition.results>` of the Composition to which they project.  If False, the outputs of `PROBE
        <NodeRole.PROBE>` Nodes *are excluded* from those attributes;  if True (the default) they are included
        (see `Probes <Composition_Probes>` for additional details).

    required_node_roles : list[(`Mechanism <Mechanism>` or `Composition`, `NodeRole`)]
        a list of tuples, each containing a `Node <Composition_Nodes>` and a `NodeRole` assigned to it.

    excluded_node_roles : list[(`Mechanism <Mechanism>` or `Composition`, `NodeRole`)]
        a list of tuples, each containing a `Node <Composition_Nodes>` and a `NodeRole` that is excluded from
        being assigned to it.

    feedback_senders : ContentAddressableList[`Node <Composition_Nodes>`]
        list of `Nodes <Composition_Nodes>` that have one or more `efferent Projections <Mechanism_Base.efferents>`
        designated as `feedback <Composition_Feedback_Designation>`.

    feedback_receivers : ContentAddressableList[`Node <Composition_Nodes>`]
        list of `Nodes <Composition_Nodes>` that have one or more `afferents <Mechanism_Base.afferents>` designated as
        `feedback <Composition_Feedback_Designation>`.

    feedback_projections : ContentAddressableList[`Projection <Projection>`]
        list of Projections in the Composition designated as `feedback <Composition_Feedback_Designation>`.

    mechanisms : `MechanismList`
        list of Mechanisms in Composition, that provides access to some of they key attributes.

    random_variables : list[Component]
        list of Components in Composition with variables that call a randomization function.

        .. technical_note::
           These are Components with a seed `Parameter`.

    pathways : ContentAddressableList[`Pathway`]
        a list of all `Pathways <Pathway>` in the Composition that were specified in the **pathways**
        argument of the Composition's constructor and/or one of its `Pathway addition methods
        <Composition_Pathway_Addition_Methods>`; each item is a list of `Nodes <Composition_Nodes>`
        (`Mechanisms <Mechanism>` and/or Compositions) intercolated with the `Projection(s) <Projection>` between each
        pair of Nodes; if both Nodes are Mechanisms, then only a single Projection can be specified;  if either is a
        Composition then, under some circumstances, there can be a set of Projections, specifying how the `INPUT
        <NodeRole.INPUT>` Node(s) of the sender project to the `OUTPUT <NodeRole.OUTPUT>` Node(s) of the receiver
        (see `add_linear_processing_pathway` for additional details).

    projections : ContentAddressableList[`Projection`]
        a list of all of the `Projections <Projection>` activated for the Composition;  this includes all of
        the Projections among `Nodes <Composition_Nodes>` within the Composition, as well as from its `input_CIM
        <Composition.input_CIM>` to it *INPUT* Nodes;from its `parameter_CIM <Composition.parameter_CIM>` to
        the corresponding `ParameterPorts <ParameterPorts>`; from its *OUTPUT* Nodes to its `output_CIM
        <Composition.output_CIM>`; and, if it is `nested <Composition_Nested>` in another Composition, then the
        Projections to its `input_CIM <Composition.input_CIM>` and from its `output_CIM <Composition.output_CIM>`
        to other Nodes in the Comopsition within which it is nested.

    input_CIM : `CompositionInterfaceMechanism`
        mediates input values for the `INPUT` `Nodes <Composition_Nodes>` of the Composition. If the Composition is
        `nested <Composition_Nested>`, then the input_CIM and its `InputPorts <InputPort> serve as proxies for the
        Composition itself for its afferent `PathwayProjections <PathwayProjection>` (see `input_CIM
        <Composition_input_CIM>` for additional details).

    input_CIM_ports : dict
        a dictionary in which the key of each entry is the `InputPort` of an `INPUT` `Node <Composition_Nodes>` in
        the Composition, and its value is a list containing two items:

        - the `InputPort` of the `input_CIM <Composition.input_CIM>` that receives the input destined for that `INPUT`
          Node -- either from the `input <Composition_Execution_Inputs>` specified for the Node in a call to one of the
          Composition's `execution methods <Composition_Execution_Methods>`, or from a `MappingProjection` from a
          Node in an `enclosing Composition <Composition_Nested>` that has specified the `INPUT` Node as its `receiver
          <Projection_Base.receiver>`;

        - the `OutputPort` of the `input_CIM <Composition.input_CIM>` that sends a `MappingProjection` to the
          `InputPort` of the `INPUT` Node.

    parameter_CIM : `CompositionInterfaceMechanism`
        mediates modulatory values for all `Nodes <Composition_Nodes>` of the Composition. If the Composition is
        `nested <Composition_Nested>`, then the parameter_CIM and its `InputPorts <InputPort>` serve as proxies for
        the Composition itself for its afferent `ModulatoryProjections <ModulatoryProjection>` (see `parameter_CIM
        <Composition_parameter_CIM>` for additional details).

    parameter_CIM_ports : dict
        a dictionary in which keys are `ParameterPorts <ParameterPort>` of `Nodes <Composition_Nodes>` in the
        Composition, and values are lists containing two items:

        - the `InputPort` of the `parameter_CIM <Composition.parameter_CIM>` that receives a `MappingProjection` from
          a `ModulatorySignal` of a `ModulatoryMechanism` in the `enclosing Composition <Composition_Nested>`;

        - the `OutputPort` of the parameter_CIM that sends a `ModulatoryProjection` to the `ParameterPort` of the Node
          in the Composition with the parameter to be modulated.

    afferents : ContentAddressableList[`Projection <Projection>`]
        a list of all of the `Projections <Projection>` to either the Composition's `input_CIM` (`PathwayProjections
        <PathwayProjection>` and `ModulatoryProjections <ModulatoryProjection>`).

    external_input_ports : list[InputPort]
        a list of the InputPorts of the Composition's `input_CIM <Composition.input_CIM>`;  these receive input
        provided to the Composition when it is `executed <Composition_Execution>`, either from the **inputs** argument
        of one of its `execution methods <Composition_Execution_Methods>` or, if it is a `nested Composition
        <Composition_Nested>`, then from any `Nodes <Composition_Nodes>` in the outer composition that project to the
        nested Composition (either itself, as a Node in the outer Composition, or to any of its own Nodes).

    external_input_ports_of_all_input_nodes : list[InputPort]
        a list of all `external InputPort <Composition_Input_External_InputPorts>` of all `INPUT <NodeRole.INPUT>`
        `Nodes <Composition_Nodes>` of the Composition, including any that in `nested Compositions
        <Composition_Nested>` within it (i.e., within `INPUT <NodeRole.INPUT>` Nodes at all levels of nesting).
        Note that the InputPorts listed are those of the actual Mechanisms projected to by the ones listed
        in `external_input_ports <Composition.external_input_ports>`.

    external_input_shape : list[1d array]
        a list of the `input_shape <InputPort.input_shape>`\\s of all of the InputPorts listed in
        `external_input_ports <Composition.external_input_ports>` (and are the same as the shapes of those listed in
        `external_input_ports_of_all_input_nodes <Composition.external_input_ports_of_all_input_nodes>`); any input
        to the Composition must be compatible with these, whether received from the **inputs** argument of one of the
        Composition's `execution methods <Composition_Execution_Methods>` or, if it is a `nested Composition
        <Composition_Nested>`, from the enclosing Composition.

    external_input_variables : list[2d array]
        a list of the `variable <InputPort.variable>`\\s associated with the `InputPorts <InputPort>` listed in
        `external_input_ports <Composition.external_input_ports>`.

    external_input_values : list[1d array]
        a list of the values of associated of the `InputPorts <InputPort>` listed in `external_input_ports
        <Composition.external_input_ports>`.

    external_input_values : list[InputPort]
        a list of the values of associated with the `InputPorts <InputPort>` listed in `external_input_ports
        <Composition.external_input_ports>`.

    output_CIM : `CompositionInterfaceMechanism`
        aggregates output values from the OUTPUT nodes of the Composition. If the Composition is nested, then the
        output_CIM and its OutputPorts serve as proxies for Composition itself in terms of efferent projections
        (see `output_CIM <Composition_output_CIM>` for additional details).

    output_CIM_ports : dict
        a dictionary in which the key of each entry is the `OutputPort` of an `OUTPUT` `Node <Composition_Nodes>` in
        the Composition, and its value is a list containing two items:

        - the `InputPort` of the output_CIM that receives a `MappingProjection` from the `OutputPort` of the `OUTPUT`
          Node;

        - the `OutputPort` of the `output_CIM <Composition.output_CIM>` that is either recorded in the `results
          <Composition.results>` attrribute of the Composition, or sends a `MappingProjection` to a Node in the
          `enclosing Composition <Composition_Nested>` that has specified the `OUTPUT` Node as its `sender
          <Projection_Base.sender>`.

    efferents : ContentAddressableList[`Projection <Projection>`]
        a list of all of the `Projections <Projection>` from the Composition's `output_CIM`.

    cims : list
        a list containing references to the Composition's `input_CIM <Composition.input_CIM>`,
        `parameter_CIM <Composition.parameter_CIM>`, and `output_CIM <Composition.output_CIM>`
        (see `Composition_CIMs` for additional details).

    env : Gym Forager Environment : default: None
        stores a Gym Forager Environment so that the Composition may interact with this environment within a
        single call to `run <Composition.run>`.

    shadows : dict
        a dictionary in which the keys are all `Nodes <Composition_Nodes>` in the Composition,
        and the values of each is a list of any Nodes that `shadow <InputPort_Shadow_Inputs>` it's input.

    controller : OptimizationControlMechanism
        identifies the `OptimizationControlMechanism` used as the Composition's controller
        (see `Composition_Controller` for details).

    enable_controller : bool
        determines whether the Composition's `controller <Composition.controller>` is executed when the Composition
        is run.  Set to True by default if `controller <Composition.controller>` is specified.  Setting it to False
        suppresses exectuion of the `controller <Composition.controller>` (see `Composition_Controller_Execution`
        for additional details, including timing of execution).

    controller_mode :  BEFORE or AFTER
        determines whether the `controller <Composition.controller>` is executed before or after the rest of the
        `Composition` when it is run, at the `TimeScale` determined by `controller_time_scale
        <Composition.controller_time_scale>` (see `Composition_Controller_Execution` for additional details).

    controller_time_scale: TimeScale[TIME_STEP, PASS, TRIAL, RUN] : default TRIAL
        deterines the frequency at which the `controller <Composition.controller>` is executed, either before or
        after the Composition as determined by `controller_mode <Composition.ontroller_mode>` (see
        `Composition_Controller_Execution` for additional details).

    controller_condition : Condition
        determines whether the `controller <Composition.controller>` is executed in a given trial.  The
        default is `Always()`, which executes the controller on every trial (see `Composition_Controller_Execution`
        for additional details).

    default_execution_id
        if no *context* is specified in a call to run, this *context* is used;  by default,
        it is the Composition's `name <Composition.name>`.

    execution_ids : set
        stores all execution_ids used by the Composition.

    disable_learning: bool : default False
        determines whether `LearningMechanisms <LearningMechanism>` in the Composition are executed when run in
        `learning mode <Composition.learn>`.

    learning_components : list[list]
        a list of the learning-related components in the Composition, all or many of which may have been
        created automatically in a call to one of its `add_<*learning_type*>_pathway' methods (see
        `Composition_Learning` for details).  This does *not* contain the `ProcessingMechanisms
        <ProcessingMechanism>` or `MappingProjections <MappingProjection>` in the pathway(s) being learned;
        those are contained in `learning_pathways <Composition.learning_pathways>` attribute.

    learned_components : list[list]
        a list of the components subject to learning in the Composition (`ProcessingMechanisms
        <ProcessingMechanism>` and `MappingProjections <MappingProjection>`);  this does *not* contain the
        components used for learning; those are contained in `learning_components
        <Composition.learning_components>` attribute.

    results : list[list[list]]
        a list of the `output_values <Mechanism_Base.output_values>` of the `OUTPUT` `Nodes <Composition_Nodes>`
        in the Composition for every `TRIAL <TimeScale.TRIAL>` executed in a call to `run <Composition.run>`.
        Each item in the outermos list is a list of values for a given trial; each item within a trial corresponds
        to the `output_values <Mechanism_Base.output_values>` of an `OUTPUT` Mechanism for that trial.

    output_values : list[list]
        a list of the `output_values <Mechanism_Base.output_values>` of the `OUTPUT` `Nodes <Composition_Nodes>`
        in the Composition for the last `TRIAL <TimeScale.TRIAL>` executed in a call to one of the Composition's
        `execution methods <Composition_Execution_Methods>`, and the value returned by that method; this is the
        same as `results <Composition.results>`\\[0], and provides consistency of access to the values of a
        Composition's Nodes when one or more is a `nested Composition <Composition_Nested>`.

    simulation_results : list[list[list]]
        a list of the `results <Composition.results>` for `simulations <OptimizationControlMechanism_Execution>`
        of the Composition when it is executed using its `evaluate <Composition.evaluate>` method by an
        `OptimizationControlMechanism`.

    retain_old_simulation_data : bool
        if True, all `Parameter <Parameters>` values generated during `simulations
        <OptimizationControlMechanism_Execution>` are saved;
        if False, simulation values are deleted unless otherwise specified by individual Parameters.

    recorded_reports : str
        contains output and/or progress reports from execution(s) of Composition if *RECORD* is specified in the
        **report_to_devices** argument of a `Composition execution method <Composition_Execution_Methods>`.

    rich_diverted_reports : str
        contains output and/or progress reports from execution(s) of Composition if *DIVERT* is specified in the
        **report_to_devices** argument of a `Composition execution method <Composition_Execution_Methods>`.

    input_specification : None or dict or list or generator or function
        stores the `inputs` for executions of the Composition when it is executed using its `run <Composition.run>`
        method.

    name : str
        the name of the Composition; if it is not specified in the **name** argument of the constructor, a default
        is assigned by CompositionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Composition; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).

    """

    # Composition now inherits from Component, so registry inherits name None
    componentType = 'Composition'
    # Set componentCategory for quick type checking of subclasses (e.g. AutodiffComposition)
    componentCategory = 'Composition'
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
        retain_old_simulation_data = Parameter(False, stateful=False, loggable=False, pnl_internal=True)
        input_specification = Parameter(None, stateful=False, loggable=False, pnl_internal=True)


    class _CompilationData(ParametersBase):
        execution = None

    @check_user_specified
    def __init__(
            self,
            pathways=None,
            nodes=None,
            projections=None,
            allow_probes:Union[bool, CONTROL]=True,
            include_probes_in_output:bool=False,
            disable_learning:bool=False,
            controller:ControlMechanism=None,
            enable_controller=None,
            controller_mode:tc.enum(BEFORE,AFTER)=AFTER,
            controller_time_scale=TimeScale.TRIAL,
            controller_condition:Condition=Always(),
            retain_old_simulation_data=None,
            show_graph_attributes=None,
            name=None,
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

        # core attributes
        self.graph = Graph()  # Graph of the Composition
        self._graph_processing = None
        self.nodes = ContentAddressableList(component_type=Component)
        self.node_ordering = []
        self.allow_probes = allow_probes
        self.include_probes_in_output=include_probes_in_output
        self.required_node_roles = []
        self.excluded_node_roles = []
        from psyneulink.core.compositions.pathway import Pathway
        self.pathways = ContentAddressableList(component_type=Pathway)

        # 'env' attr required for dynamic inputs generated by gym forager env
        self.env = None

        self.input_CIM_ports = {}
        self.parameter_CIM_ports = {}
        self.output_CIM_ports = {}

        # Interface Mechanisms
        self.input_CIM = CompositionInterfaceMechanism(name=self.name + " Input_CIM",
                                                       composition=self,
                                                       port_map=self.input_CIM_ports)
        self.parameter_CIM = CompositionInterfaceMechanism(name=self.name + " Parameter_CIM",
                                                        composition=self,
                                                       port_map=self.parameter_CIM_ports)
        self.output_CIM = CompositionInterfaceMechanism(name=self.name + " Output_CIM",
                                                        composition=self,
                                                        port_map=self.output_CIM_ports)
        self.cims = [self.input_CIM, self.parameter_CIM, self.output_CIM]

        self.default_execution_id = self.name
        self.execution_ids = {self.default_execution_id}

        self.projections = ContentAddressableList(component_type=Component)

        self._scheduler = None
        self._partially_added_nodes = []

        self.disable_learning = disable_learning

        # graph and scheduler status attributes
        self.graph_consistent = True  # Tracks if Composition is in runnable state (no dangling projections (what else?)
        self.needs_update_graph = True  # Tracks if Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True  # Tracks if the processing graph is current with the full graph
        self.needs_update_scheduler = True  # Tracks if the scheduler needs to be regenerated
        self.needs_update_controller = True # Tracks if controller needs to update its state_input_ports
        self.needs_determine_node_roles = False # Set in add_node and add_projection to insure update of NodeRoles
        self._need_check_for_unused_projections = True

        self.nodes_to_roles = collections.OrderedDict()
        self.cycle_vertices = set()

        context = Context(source=ContextFlags.CONSTRUCTOR, execution_id=None)

        self._initialize_parameters(
            **param_defaults,
            retain_old_simulation_data=retain_old_simulation_data,
            context=context
        )

        # Compiled resources
        self._compilation_data = self._CompilationData(owner=self)

        # If a PreferenceSet was provided, assign to instance
        _assign_prefs(self, prefs, BasePreferenceSet)

        self.log = CompositionLog(owner=self)
        self._terminal_backprop_sequences = {}

        self.controller = None

        # FIX 4/8/20 [JDC]: WHY NOT CALL add_nodes()?
        # Nodes, Projections, and Pathways
        if nodes is not None:
            nodes = convert_to_list(nodes)
            for node in nodes:
                required_roles = None
                if isinstance(node, tuple):
                    node, required_roles = node
                self.add_node(node, required_roles)

        # FIX 4/8/20 [JDC]: TEST THIS
        if projections is not None:
            projections = convert_to_list(projections)
            self.add_projections(projections)

        self.add_pathways(pathways, context=context)

        # Controller
        self.controller = None
        self._controller_initialization_status = ContextFlags.INITIALIZED
        self.enable_controller = enable_controller
        if controller:
            self.add_controller(controller)
        self.controller_mode = controller_mode
        self.controller_time_scale = controller_time_scale
        self.controller_condition = controller_condition
        self.controller_condition.owner = self.controller
        # This is set at runtime and may be used by the controller to assign its
        #     `num_trials_per_estimate <OptimizationControlMechanism.num_trials_per_estimate>` attribute.
        self.num_trials = None

        self._update_parameter_components()

        self.initialization_status = ContextFlags.INITIALIZED
        #FIXME: This removes `composition.parameters.values`, as it was not being
        # populated correctly in the first place. `composition.parameters.results`
        # should be used instead - in the long run, we should look into possibly
        # populating both values and results, as it would be more consistent with
        # the behavior of components
        del self.parameters.value

        # Call with context = COMPOSITION to avoid calling _check_initialization_status again
        self._analyze_graph(context=context)

        show_graph_attributes = show_graph_attributes or {}
        self._show_graph = ShowGraph(self, **show_graph_attributes)

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
            when it is `run <Composition_Execution>`.

            :getter: Returns the default scheduler, and builds it if it needs updating since the last access.
        """
        if self.needs_update_scheduler or not isinstance(self._scheduler, Scheduler):
            old_scheduler = self._scheduler
            if old_scheduler is not None:
                orig_conds = old_scheduler._user_specified_conds
            else:
                orig_conds = None

            self._scheduler = Scheduler(composition=self, conditions=orig_conds)
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

    @property
    def scheduling_mode(self):
        return self.scheduler.scheduling_mode

    @scheduling_mode.setter
    def scheduling_mode(self, scheduling_mode: SchedulingMode):
        self.scheduler.scheduling_mode = scheduling_mode

    # ******************************************************************************************************************
    # region -------------------------------------- GRAPH  -------------------------------------------------------------
    # ******************************************************************************************************************

    @handle_external_context(source=ContextFlags.COMPOSITION)
    def _analyze_graph(self, context=None):
        """
        Assigns `NodeRoles <NodeRole>` to nodes based on the structure of the `Graph`.

        By default, if _analyze_graph determines that a Node is `ORIGIN <NodeRole.ORIGIN>`, it is also given the role
        `INPUT <NodeRole.INPUT>`. Similarly, if _analyze_graph determines that a Node is `TERMINAL
        <NodeRole.TERMINAL>`, it is also given the role `OUTPUT <NodeRole.OUTPUT>`.

        However, if the **required_roles** argument of `add_node <Composition.add_node>` is used to set any Node in the
        Composition to `INPUT <NodeRole.INPUT>`, then the `ORIGIN <NodeRole.ORIGIN>` nodes are not set to `INPUT
        <NodeRole.INPUT>` by default. If the **required_roles** argument of `add_node <Composition.add_node>` is used
        to set any Node in the Composition to `OUTPUT <NodeRole.OUTPUT>`, then the `TERMINAL <NodeRole.TERMINAL>`
        nodes are not set to `OUTPUT <NodeRole.OUTPUT>` by default.
        """

        self._check_controller_initialization_status(context=context)
        self._check_nodes_initialization_status(context=context)

        # FIX: SHOULDN'T THIS TEST MORE EXPLICITLY IF NODE IS A Composition?
        # Call _analzye_graph() for any nested Compositions
        for n in self.nodes:
            try:
                n._analyze_graph(context=context)
            except AttributeError:
                pass

        self._complete_init_of_partially_initialized_nodes(context=context)
        # Call before _determine_pathway and _create_CIM_ports so they have updated roles
        self._determine_node_roles(context=context)
        self._determine_pathway_roles(context=context)
        self._create_CIM_ports(context=context)
        # Call after above so shadow_projections have relevant organization
        self._update_shadow_projections(context=context)
        self._check_for_projection_assignments(context=context)
        self.needs_update_graph = False

    def _update_processing_graph(self):
        """
        Constructs the processing graph (the graph that contains only Nodes as vertices)
        from the composition's full graph
        """
        self._graph_processing = self.graph.copy()

        def remove_vertex(vertex):
            for parent in vertex.parents:
                for child in vertex.children:
                    child.source_types[parent] = vertex.feedback
                    self._graph_processing.connect_vertices(parent, child)

            self._graph_processing.remove_vertex(vertex)

        # copy to avoid iteration problems when deleting
        vert_list = self._graph_processing.vertices.copy()
        for cur_vertex in vert_list:
            if not cur_vertex.component.is_processing:
                remove_vertex(cur_vertex)

        # this determines CYCLE nodes and final FEEDBACK nodes
        self._graph_processing.prune_feedback_edges()
        self.needs_update_graph_processing = False

    # endregion GRAPH

    # ******************************************************************************************************************
    # region ---------------------------------------NODES  -------------------------------------------------------------
    # ******************************************************************************************************************

    @handle_external_context(source = ContextFlags.COMPOSITION)
    def add_node(self, node, required_roles=None, context=None):
        """
            Add a Node (`Mechanism <Mechanism>` or `Composition`) to Composition, if it is not already added

            Arguments
            ---------

            node : `Mechanism <Mechanism>` or `Composition`
                the Node to be added to the Composition

            required_roles : `NodeRole` or list of NodeRoles
                any NodeRoles roles that this Node should have in addition to those determined by analyze graph.
        """

        # FIX 5/25/20 [JDC]: ADD ERROR STRING (as in pathway_arg_str in add_linear_processing_pathway)
        # Raise error if Composition is added to itself
        if node is self:
            pathway_arg_str = ""
            if context.source in {ContextFlags.INITIALIZING, ContextFlags.METHOD}:
                pathway_arg_str = " in " + context.string
            raise CompositionError(f"Attempt to add Composition as a Node to itself{pathway_arg_str}.")

        required_roles = convert_to_list(required_roles)

        if isinstance(node, Composition):
            # IMPLEMENTATION NOTE: include_probes_in_output=False is not currently supported for nested Nodes
            #                    (they require get_output_value() to return value of all output_ports of output_CIM)
            node.include_probes_in_output = True
        else:
            if required_roles and NodeRole.INTERNAL in required_roles:
                for input_port in node.input_ports:
                    input_port.internal_only = True
        try:
            node._analyze_graph(context = context)
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
            self.needs_update_controller = True

        invalid_aux_components = self._add_node_aux_components(node)

        # Implement required_roles
        if required_roles:
            if not isinstance(required_roles, list):
                required_roles = [required_roles]
            for required_role in required_roles:
                self._add_required_node_role(node, required_role, context)

        # Add ControlSignals to controller and ControlProjections
        #     to any parameter_ports specified for control in node's constructor
        if self.controller:
            self._instantiate_deferred_init_control(node, context=context)

        try:
            if len(invalid_aux_components) > 0:
                self._partially_added_nodes.append(node)
        except NameError:
            pass

        if isinstance(node, ControlMechanism):
            self._handle_allow_probes_for_control(node)

        self._need_check_for_unused_projections = True

        # # MODIFIED 1/27/22 NEW - FIX - BREAKS test_learning_output_shape() in ExecuteMode.LLVM
        #                                [3/25/22] STILL NEEDED (e.g., FOR test_inputs_key_errors()
        # if context.source != ContextFlags.METHOD:
        #     # Call _analyze_graph with ContextFlags.METHOD to avoid recursion
        #     self._analyze_graph(context=Context(source=ContextFlags.METHOD))
        # MODIFIED 1/27/22 END
        self.needs_determine_node_roles = True

    def add_nodes(self, nodes, required_roles=None, context=None):
        """
            Add a list of `Nodes <Composition_Nodes>` to the Composition.

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
            nodes = convert_to_list(nodes)
        for node in nodes:
            if isinstance(node, (Mechanism, Composition)):
                self.add_node(node, required_roles, context)
            elif isinstance(node, tuple):
                node_specific_roles = convert_to_list(node[1])
                if required_roles:
                    node_specific_roles.append(required_roles)
                self.add_node(node=node[0], required_roles=node_specific_roles, context=context)
            else:
                raise CompositionError(f"Node specified in 'add_nodes' method of '{self.name}' {Composition.__name__} "
                                       f"({node}) must be a {Mechanism.__name__}, {Composition.__name__}, "
                                       f"or a tuple containing one of those and a {NodeRole.__name__} or list of them")

    def remove_node(self, node):
        self._remove_node(node)

    def _remove_node(self, node, analyze_graph=True):
        for proj in node.afferents + node.efferents:
            self.remove_projection(proj)

        for param_port in node.parameter_ports:
            for proj in param_port.mod_afferents:
                self.remove_projection(proj)

        # deactivate any shadowed projections
        for shadow_target, shadow_port_original in self.shadowing_dict.items():
            if shadow_port_original in node.input_ports:
                for shadow_proj in shadow_target.all_afferents:
                    if shadow_proj.sender.owner.composition is self:
                        self.remove_projection(shadow_proj)

                        # NOTE: deactivation should be sufficient but
                        # asserts in OCM _update_state_input_port_names
                        # need target input ports of shadowed
                        # projections to be active or not present at all
                        try:
                            self.controller.state_input_ports.remove(shadow_target)
                        except AttributeError:
                            pass

        self.graph.remove_component(node)
        del self.nodes_to_roles[node]

        # Remove any entries for node in required_node_roles or excluded_node_roles
        node_role_pairs = [item for item in self.required_node_roles if item[0] is node]
        for item in node_role_pairs:
            self.required_node_roles.remove(item)
        node_role_pairs = [item for item in self.excluded_node_roles if item[0] is node]
        for item in node_role_pairs:
            self.excluded_node_roles.remove(item)

        del self.nodes[node]
        self.node_ordering.remove(node)

        for p in self.pathways:
            try:
                p.pathway.remove(node)
            except ValueError:
                pass

        self.needs_update_graph_processing = True
        self.needs_update_scheduler = True

        if analyze_graph:
            self._analyze_graph()

    def remove_nodes(self, nodes):
        if not isinstance(nodes, (list, Mechanism, Composition)):
            assert False, 'Argument of remove_nodes must be a Mechanism, Composition or list containing either or both'
        nodes = convert_to_list(nodes)
        for node in nodes:
            self._remove_node(node, analyze_graph=False)

        self._analyze_graph()

    @handle_external_context()
    def _add_required_node_role(self, node, role, context=None):
        """
            Assign the `NodeRole` specified by **role** to **node**.  Remove exclusion of that `NodeRole` if
            it had previously been specified in `exclude_node_roles <Composition.exclude_node_roles>`.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` to which **role** should be assigned.

            role : `NodeRole`
                `NodeRole` to assign to **node**.

        """
        if role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        # Disallow assignment of NodeRoles by user that are not programmitically modifiable:
        # FIX 4/25/20 [JDC]: CHECK IF ROLE OR EQUIVALENT STATUS HAS ALREADY BEEN ASSIGNED AND, IF SO, ISSUE WARNING
        if context.source == ContextFlags.COMMAND_LINE:
            if role in {NodeRole.CONTROL_OBJECTIVE, NodeRole.CONTROLLER_OBJECTIVE}:
                # raise CompositionError(f"{role} cannot be directly assigned to an {ObjectiveMechanism.__name__};"
                #                        # f"assign 'CONTROL' to 'role' argument of consructor for {node} of {self.name}")
                #                        f"try assigning {node} to '{OBJECTIVE_MECHANISM}' argument of "
                #                        f"the constructor for the desired {ControlMechanism.__name__}.")
                warnings.warn(f"{role} should be assigned with caution to {self.name}. "
                              f"{ObjectiveMechanism.__name__}s are generally constructed automatically by a "
                              f"{ControlMechanism.__name__}, or assigned to it in the '{OBJECTIVE_MECHANISM}' "
                              f"argument of its constructor.  Doing so otherwise may cause unexpected results.")
            elif role in {NodeRole.LEARNING, NodeRole.LEARNING_OBJECTIVE, NodeRole.TARGET}:
                warnings.warn(f"{role} should be assigned with caution to {self.name}. "
                              f"Learning Components are generally constructed automatically as part of "
                              f"a learning Pathway. Doing so otherwise may cause unexpected results.")
            elif role in {NodeRole.FEEDBACK_SENDER, NodeRole.FEEDBACK_RECEIVER}:
                to_from = 'from'
                if role is NodeRole.FEEDBACK_RECEIVER:
                    to_from = 'to'
                from psyneulink.core.components.projections.projection import Projection
                warnings.warn(f"{role} is not a role that can be assigned directly {to_from} {self.name}. "
                              f"The relevant {Projection.__name__} to it must be designated as 'feedback' "
                              f"where it is addd to the {self.name};  assignment will be ignored.")
            elif role in {NodeRole.ORIGIN, NodeRole.INTERNAL, NodeRole.SINGLETON, NodeRole.TERMINAL, NodeRole.CYCLE}:
                raise CompositionError(f"Attempt to assign {role} (to {node} of {self.name})"
                                       f"that cannot be modified by user.")

        node_role_pair = (node, role)
        if node_role_pair not in self.required_node_roles:
            self.required_node_roles.append(node_role_pair)
        node_role_pairs = [item for item in self.excluded_node_roles if item[0] is node and item[1] is role]
        for item in node_role_pairs:
            self.excluded_node_roles.remove(item)

    @handle_external_context()
    def require_node_roles(self, node, roles, context=None):
        """
            Assign the `NodeRole`\\(s) specified in **roles** to **node**.  Remove exclusion of those NodeRoles if
            it any had previously been specified in `exclude_node_roles <Composition.exclude_node_roles>`.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` to which **role** should be assigned.

            roles : `NodeRole` or list[`NodeRole`]
                `NodeRole`\\(s) to assign to **node**.

        """
        roles = convert_to_list(roles)
        for role in roles:
            self._add_required_node_role(node, role, context)

    @handle_external_context()
    def exclude_node_roles(self, node, roles, context):
        """
            Excludes the `NodeRole`\\(s) specified in **roles** from being assigned to **node**.

            Removes specified roles if they had been previous assigned either by default as a `required_node_role
            <Composition_Node_Role_Assignment>` or using the `required_node_roles <Composition.required_node_roles>`
            method.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` from which **role** should be removed.

            roles : `NodeRole` or list[`NodeRole`]
                `NodeRole`\\(s) to remove and/or exclude from **node**.

        """
        roles = convert_to_list(roles)

        for role in roles:
            if role not in NodeRole:
                raise CompositionError(f"Invalid NodeRole specified for {node} in 'exclude_node_roles': {role}.")

            # Disallow assignment of NodeRoles by user that are not programmitically modifiable:
            if (context.source == ContextFlags.COMMAND_LINE and
                    role in {NodeRole.ORIGIN, NodeRole.INTERNAL, NodeRole.SINGLETON, NodeRole.TERMINAL,
                             NodeRole.CYCLE, NodeRole.FEEDBACK_SENDER, NodeRole.FEEDBACK_RECEIVER, NodeRole.LEARNING}):
                raise CompositionError(f"Attempt to exclude {role} (from {node} of {self.name})"
                                       f"that cannot be modified by user.")
            node_role_pair = (node, role)
            self.excluded_node_roles.append(node_role_pair)
            if node_role_pair in self.required_node_roles:
                self.required_node_roles.remove(node_role_pair)
            self._remove_node_role(node, role)

    def get_roles_by_node(self, node):
        """
            Return a list of `NodeRoles <NodeRole>` assigned to **node**.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` for which assigned `NodeRoles <NodeRole>` are desired.

            Returns
            -------

            List[`Mechanisms <Mechanism>` and/or `Compositions <Composition>`] :
                list of `NodeRoles <NodeRole>` assigned to **node**.
        """

        try:
            return self.nodes_to_roles[node]
        except KeyError:
            raise CompositionError(f"Node {node} not found in {self.nodes_to_roles}.")

    def get_nodes_by_role(self, role):
        """
            Return a list of `Nodes <Composition_Nodes>` in the Composition that are assigned the `NodeRole`
            specified in **role**.

            Arguments
            _________

            role : `NodeRole`
                role for which `Nodes <Composition_Nodes>` are desired.

            Returns
            -------

            list[`Mechanisms <Mechanism>` and/or `Compositions <Composition>`] :
                list of `Nodes <Composition_Nodes>` assigned the `NodeRole` specified in **role**

        """
        if role is None or role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        try:
            return [node for node in self.nodes_to_roles if role in self.nodes_to_roles[node]]

        except KeyError as e:
            raise CompositionError('Node missing from {0}.nodes_to_roles: {1}'.format(self, e))

    def _get_nested_nodes_with_same_roles_at_all_levels(self, comp, include_roles, exclude_roles=None):
        """Return all Nodes from nested Compositions that have *include_roles* but not *exclude_roles at all levels*.
        Note:  need to do this recursively, checking roles on the "way down," since a Node may have a role in a
               deeply nested Composition, but that Composition itself may not have the same role in the Composition
               within which *it* is nested (e.g., a Node might be an INPUT Node of a nested Composition, but that
               nested Composition may not be an INPUT Node of the Composition in which it is nested).
        """
        nested_nodes = []
        include_roles = convert_to_list(include_roles)
        if exclude_roles:
            exclude_roles = convert_to_list(exclude_roles)
        else:
            exclude_roles = []
        if isinstance(comp, Composition):
            # Get all nested nodes in comp that have include_roles and not exclude_roles:
            for node in [n for n in comp.nodes
                         if (any(n in comp.get_nodes_by_role(include)
                                 for include in include_roles)
                               and not any(n in comp.get_nodes_by_role(exclude)
                                           for exclude in exclude_roles))]:
                if isinstance(node, Composition):
                    nested_nodes.extend(node._get_nested_nodes_with_same_roles_at_all_levels(node, include_roles,
                                                                                             exclude_roles))
                else:
                    nested_nodes.append(node)
        return nested_nodes or None

    def _get_input_nodes_by_CIM_input_order(self):
        """Return a list with the `INPUT` `Nodes <Composition_Nodes>` of the Composition in the same order as their
           corresponding InputPorts on Composition's `input_CIM <Composition.input_CIM>`.
        """

        return [{cim[0]:n for n, cim in self.input_CIM_ports.items()}[input_port].owner
                for input_port in self.input_CIM.input_ports]

    def _get_input_receivers(self,
                             comp=None,
                             type:Union[PORT,NODE]=PORT,
                             comp_as_node:Union[bool,ALL]=False):
        """Return all INPUT Nodes [or their InputPorts] of comp, [including those for any nested Compositions].
        If type is PORT, return all InputPorts for all INPUT Nodes, including for nested Compositions.
        If type is NODE, return all INPUT Nodes, including for nested Compositions as determined by comp_as_node:
            if an INPUT Node is a Composition, and comp_as_node is:
            - False, include the nested Composition's INPUT Nodes, but not the Composition
            - True, include the nested Composition but not its INPUT Nodes
            - ALL, include the nested Composition AND its INPUT Nodes
        """

        # FIX: 3/16/22 - CAN THIS BE REPLACED BY:
        # return [self._get_destination(output_port.efferents[0])[0]
        #         for _,output_port in self.input_CIM.port_map.values()]

        assert not (type == PORT and comp_as_node), f"PROGRAM ERROR: _get_input_receivers() can't be called " \
                                                    f"for 'ports' and 'nodes' at the same time."
        comp = comp or self
        input_items = []
        _update_cim = False

        if comp.needs_determine_node_roles:
            comp._determine_node_roles()
            _update_cim = True

        if type==PORT:
            # Return all InputPorts of all INPUT Nodes
            _input_nodes = comp._get_nested_nodes_with_same_roles_at_all_levels(comp=comp,
                                                                                include_roles=NodeRole.INPUT)
            if _input_nodes:
                for node in _input_nodes:
                    # Exclude internal_only and shadowers of input (since the latter get inputs for the shadowed item)
                    input_items.extend([input_port for input_port in node.input_ports
                                        if not (input_port.internal_only or input_port.shadow_inputs)])
                # Ensure correct number of InputPorts have been identified
                #    (i.e., number of InputPorts on comp's input_CIM)
                if _update_cim:
                    context = Context()
                    self._determine_pathway_roles(context=context)
                    self._determine_pathway_roles(context)
                    self._create_CIM_ports(context)
                _update_cim = False
                assert len(input_items) == len(comp.input_CIM_ports)
        else:
            # Return all INPUT Nodes
            _input_nodes = comp.get_nodes_by_role(NodeRole.INPUT)
            for node in _input_nodes:
                if isinstance(node, Composition):
                    if comp_as_node:
                        input_items.append(node)
                    if comp_as_node in {False, ALL}:
                        input_items.extend(self._get_input_receivers(comp=node, type=type, comp_as_node=comp_as_node))
                else:
                    input_items.append(node)

        return input_items

    def _get_external_cim_input_port(self, port:InputPort, outer_comp=None):
        """Get input_CIM.input_port of outer(most) Composition that projects to port in nested Composition.
        **port** must be an InputPort that receives a single path_afferent Projection from an input_CIM.
        Search up nesting hierarchy to find the input_CIM of comp nested in outer_comp or of the outermost Composition.
        Return tuple with:
            input_CIM.input_port of the input_CIM of comp nested in outer_comp or of the outermost Composition
            input_CIM.output_port corresponding to input_CIM.input_port (for ease of tracking Node to which it projects)
        """
        assert len(port.path_afferents) == 1, \
            f"PROGRAM ERROR: _get_external_cim_input_ports called for {port} " \
            f"that has either no or more than one path_afferent projections."
        input_CIM = port.path_afferents[0].sender.owner
        assert isinstance(input_CIM, CompositionInterfaceMechanism), \
            f"PROGRAM ERROR: _get_external_cim_input_ports called for {port} that is not an INPUT Node " \
            f"(i.e., does not receive a path_afferent projection from an input_CIM of its enclosing Composition."
        input_CIM_input_port = input_CIM.port_map[port][0]
        if (outer_comp and input_CIM.composition in outer_comp.nodes) or not input_CIM_input_port.path_afferents:
            # input_CIM_input_port belongs to outermost Composition, so return
            return input_CIM.port_map[port]
        # input_CIM_input_port belongs to a nested Composition, so continue to search up the nesting hierarchy
        return self._get_external_cim_input_port(input_CIM_input_port, outer_comp)

    def _get_nested_nodes(self,
                          nested_nodes=NotImplemented,
                          root_composition=NotImplemented,
                          visited_compositions=NotImplemented):
        """Recursively search and return all nodes of all nested Compositions
           in a tuple with Composition in which they are nested.
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

    def _handle_allow_probes_for_control(self, node):
        """Reconcile allow_probes for Composition and any ControlMechanisms assigned to it, including controller.
        """
        assert isinstance(node, ControlMechanism), \
            f"PROGRAM ERROR: Attempt to handle 'allow_probes' arg for non-ControlMechanism."
        # If ControlMechanism has specified allow_probes, assign at least CONTROL to Composition.allow_probes
        if not self.allow_probes and node.allow_probes:
            self.allow_probes = CONTROL
        # If allow_probes is specified on Composition as CONTROL, then turn it on for ControlMechanism
        node.allow_probes = node.allow_probes or self.allow_probes is CONTROL

    def _get_nested_compositions(self,
                                 nested_compositions=NotImplemented,
                                 visited_compositions=NotImplemented):
        """Recursively search for and return all nested compositions.

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

    def _get_all_nodes(self):
        """Return all nodes, including those within nested Compositions at any level
        Note:  this is distinct from the _all_nodes property, which returns all nodes at the top level
        """
        return [k[0] for k in self._get_nested_nodes()] + list(self.nodes)

    def _is_in_composition(self, component, nested=True):
        """Return True if component is in Composition, including any nested Compositions if **nested** is True
        Include input_CIM and output_CIM for self and all nested Compositions
        """
        if isinstance(component, Port):
            component = component.owner

        if component in self._all_nodes:
            return True
        if nested:
            return any(component in comp._all_nodes for comp in self._get_nested_compositions())

    def _determine_origin_and_terminal_nodes_from_consideration_queue(self):
        """Assigns NodeRole.ORIGIN to all nodes in the first entry of the consideration queue and NodeRole.TERMINAL
           to all nodes in the last entry of the consideration queue. The ObjectiveMechanism of a Composition's
           controller may not be NodeRole.TERMINAL, so if the ObjectiveMechanism is the only node in the last entry
           of the consideration queue, then the second-to-last entry is NodeRole.TERMINAL instead.
        """
        queue = self.scheduler.consideration_queue

        for node in list(queue)[0]:
            self._add_node_role(node, NodeRole.ORIGIN)

        for node in list(queue)[-1]:
            if NodeRole.CONTROLLER_OBJECTIVE not in self.get_roles_by_node(node):
                self._add_node_role(node, NodeRole.TERMINAL)
            elif len(queue[-1]) < 2:
                for previous_node in queue[-2]:
                    self._add_node_role(previous_node, NodeRole.TERMINAL)

        # IMPLEMENTATION NOTE:
        #   The following is needed because the assignments above only identify nodes in the *last* consideration_set;
        #   however, the TERMINAL node(s) of a pathway with fewer nodes than the longest one may not be in the last
        #   consideration set.  Identifying these assumes that graph_processing has been called/updated,
        #   which identifies and "breaks" cycles, and assigns FEEDBACK_SENDER to the appropriate consideration set(s).
        for node in self.nodes:
            if not any([
                efferent.is_active_in_composition(self) for efferent in node.efferents
                if efferent.receiver.owner is not self.output_CIM
            ]):
                self._add_node_role(node, NodeRole.TERMINAL)

    def _add_node_aux_components(self, node, context=None):
        """Add aux_components of node to Composition.

        Returns
        -------
        list containing references to all invalid aux components
        """

        invalid_aux_components = []
        if hasattr(node, "aux_components"):
            # Collect the node's aux components that are not currently able to be added to the Composition;
            # ignore these for now and try to activate them again during every call to _analyze_graph
            # and, at runtime, if there are still any invalid aux_components left, issue a warning
            projections = []
            # Add all "nodes" to the composition first (in case projections reference them)
            for i, component in enumerate(node.aux_components):
                if isinstance(component, (Mechanism, Composition)):
                    if isinstance(component, Composition):
                        component._analyze_graph()
                    self.add_node(component)
                elif isinstance(component, Projection):
                    proj_tuple = (component, False)
                    projections.append(proj_tuple)
                    node.aux_components[i] = proj_tuple
                elif isinstance(component, tuple):
                    if isinstance(component[0], Projection):
                        if (isinstance(component[1], bool) or component[1] in {EdgeType.FLEXIBLE, MAYBE}):
                            projections.append(component)
                        else:
                            raise CompositionError("Invalid Component specification ({}) in {}'s aux_components. If a "
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
                                raise CompositionError("Invalid Component specification ({}) in {}'s aux_components. "
                                                       "If a tuple is used to specify a Mechanism or Composition, then "
                                                       "the index 0 item must be the Node, and the index 1 item must "
                                                       "be the required_roles".format(component, node.name))

                        else:
                            raise CompositionError("Invalid Component specification ({}) in {}'s aux_components. If a "
                                                   "tuple is used to specify a Mechanism or Composition, then the "
                                                   "index 0 item must be the Node, and the index 1 item must be the "
                                                   "required_roles".format(component, node.name))
                    else:
                        raise CompositionError("Invalid Component specification ({}) in {}'s aux_components. If a tuple"
                                               " is specified, then the index 0 item must be a Projection, Mechanism, "
                                               "or Composition.".format(component, node.name))
                else:
                    raise CompositionError("Invalid Component ({}) in {}'s aux_components. Must be a Mechanism, "
                                           "Composition, Projection, or tuple."
                                           .format(component.name, node.name))

            invalid_aux_components.extend(self._get_invalid_aux_components(node))

            # Add all Projections to the Composition
            for proj_spec in [i for i in projections if not i[0] in invalid_aux_components]:
                # The proj_spec assumes a direct connection between sender and receiver, and is therefore invalid if
                # either are nested (i.e. projections between them need to be routed through a CIM). In these cases,
                # a new projection is instantiated between sender and receiver instead of using the original spec.
                # If the sender or receiver is an AutoAssociativeProjection, then the owner will be another projection
                # instead of a mechanism, so owner_mech instead needs to be used instead.
                sender_node = proj_spec[0].sender.owner
                receiver_node = proj_spec[0].receiver.owner
                if isinstance(sender_node, AutoAssociativeProjection):
                    sender_node = proj_spec[0].sender.owner.owner_mech
                if isinstance(receiver_node, AutoAssociativeProjection):
                    receiver_node = proj_spec[0].receiver.owner.owner_mech
                if sender_node in self._all_nodes and \
                        receiver_node in self._all_nodes:
                    self.add_projection(projection=proj_spec[0],
                                        feedback=proj_spec[1])
                else:
                    self.add_projection(sender=proj_spec[0].sender,
                                        receiver=proj_spec[0].receiver,
                                        feedback=proj_spec[1])
                del node.aux_components[node.aux_components.index(proj_spec)]

            # MODIFIED 12/29/21 NEW:
            # # Finally, check for any deferred_init Projections
            invalid_aux_components.extend([p for p in node.projections
                                           if p._initialization_status & ContextFlags.DEFERRED_INIT])
            # MODIFIED 12/29/21 END

        return invalid_aux_components

    def _get_invalid_aux_components(self, node):
        """
        Return any Components in aux_components for a node that references items not (yet) in this Composition
        """
        # FIX 11/20/21: THIS APPEARS TO ONLY HANDLE PROJECTIONS AND NOT COMPOSITIONS OR MECHANISMS
        #  (OTHER THAN THE COMPOSITION'S controller AND ITS objective_mechanism)

        # First get all valid nodes:
        # - nodes in Composition
        # - nodes in any nested Compositions
        # - controller and associated objective_mechanism
        valid_nodes = [node for node in self.nodes.data] + \
                      [node for node, composition in self._get_nested_nodes()] + \
                      [self]
        if self.controller:
            valid_nodes.append(self.controller)
            if hasattr(self.controller,'objective_mechanism'):
                valid_nodes.append(self.controller.objective_mechanism)

        # Then get invalid components:
        #   - Projections that have senders or receivers not in the Composition
        #     (this includes any in aux_components of node, or associated with any Mechanism listed in aux_components)
        invalid_components = []
        for aux in node.aux_components:
            component = None
            if isinstance(aux, Projection):
                component = aux
            elif hasattr(aux, '__iter__'):
                for i in aux:
                    if isinstance(i, Projection):
                        component = i
                    elif isinstance(i, Mechanism):
                        if self._get_invalid_aux_components(i):
                            invalid_components.append(i)
            elif isinstance(aux, Mechanism):
                if self._get_invalid_aux_components(aux):
                    invalid_components.append(aux)
            if not component:
                continue
            if isinstance(component, Projection):
                if hasattr(component.sender, OWNER_MECH):
                    sender_node = component.sender.owner_mech
                else:
                    if isinstance(component.sender.owner, CompositionInterfaceMechanism):
                        sender_node = component.sender.owner.composition
                    else:
                        sender_node = component.sender.owner
                if hasattr(component.receiver, OWNER_MECH):
                    receiver_node = component.receiver.owner_mech
                else:
                    if isinstance(component.receiver.owner, CompositionInterfaceMechanism):
                        receiver_node = component.receiver.owner.composition
                    else:
                        receiver_node = component.receiver.owner
                # Defer instantiation of all shadow Projections until call to _update_shadow_projections()
                if (not all([sender_node in valid_nodes, receiver_node in valid_nodes])
                        or (hasattr(component.receiver, SHADOW_INPUTS) and component.receiver.shadow_inputs)):
                    invalid_components.append(component)
        if invalid_components:
            return invalid_components
        else:
            return []

    def _complete_init_of_partially_initialized_nodes(self, context=None):
        """
        Attempt to complete initialization of aux_components for any nodes with
            aux_components that were not previously compatible with Composition
        """
        completed_nodes = []
        for node in self._partially_added_nodes:
            invalid_aux_components = self._add_node_aux_components(node, context=context)
            if not invalid_aux_components:
                completed_nodes.append(node)
        self._partially_added_nodes = list(set(self._partially_added_nodes) - set(completed_nodes))

        if self.controller:
            # Avoid unnecessary updating on repeated calls to run()
            if self.needs_update_controller and hasattr(self.controller, 'state_input_ports'):
                self.controller._update_state_input_ports_for_controller(context=context)

            # Make sure all is in order at run time
            if context.flags & ContextFlags.PREPARING:
                self.controller._validate_monitor_for_control(self._get_all_nodes())
                self._instantiate_control_projections(context=context)

    def _determine_node_roles(self, context=None):
        """Assign NodeRoles to Nodes in Composition

        .. note::
           Assignments are **not** subject to user-modification (i.e., "programmatic assignment")
           unless otherwise noted.

        Assignment criteria:

        ORIGIN:
          - all Nodes that are in first consideration_set (i.e., self.scheduler.consideration_queue[0]).
          .. _note::
             - this takes account of any Projections designated as feedback by graph_processing
               (i.e., self.graph.comp_to_vertex[efferent].feedback == EdgeType.FEEDBACK)
             - these will all be assigined afferent Projections from Composition.input_CIM

        INPUT:
          - all ORIGIN Nodes for which INPUT has not been removed and/or excluded using exclude_node_roles();
          - all Nodes for which INPUT has been assigned as a required_node_role by user
            (i.e., in self.required_node_roles[NodeRole.INPUT].

        SINGLETON:
          - all Nodes that are *both* ORIGIN and TERMINAL

        INTERNAL:
          - all Nodes that are *neither* ORIGIN nor TERMINAL

        CYCLE:
          - all Nodes that identified as being in a cycle by self.graph_processing
            (i.e., in self.graph_processing.cycle_vertices)

        FEEDBACK_SENDER:
          - all Nodes that send a Projection designated as feedback by self.graph_processing OR
            specified as feedback by user

        FEEDBACK_RECEIVER:
          - all Nodes that receive a Projection designated as feedback by self.graph_processing OR
            specified as feedback by user

        CONTROL_OBJECTIVE
          - ObjectiveMechanism assigned CONTROL_OBJECTIVE as a required_node_role in ControlMechanism's
            _instantiate_objective_mechanism()
          .. note::
             - *not the same as* CONTROLLER_OBJECTIVE
             - all project to a ControlMechanism

        CONTROLLER_OBJECTIVE
          - ObjectiveMechanism assigned CONTROLLER_OBJECTIVE as a required_node_role in add_controller()
          .. note::
             - also assigned CONTROL_OBJECTIVE
             - *not the same as* CONTROL_OBJECTIVE

        LEARNING
          - all Nodes for which LEARNING is assigned as a required_noded_role in
            add_linear_learning_pathway() or _create_terminal_backprop_learning_components()

        TARGET
          - all Nodes for which TARGET has been assigned as a required_noded_role in
            add_linear_learning_pathway() or _create_terminal_backprop_learning_components()
          .. note::
             - receive a Projection from input_CIM, and project to LEARNING_OBJECTIVE and output_CIM
             - also assigned ORIGIN, INPUT, LEARNING, OUTPUT, and TERMINAL

        LEARNING_OBJECTIVE
          - all Nodes for which LEARNING_OBJECTIVE is assigned required_noded_role in
            add_linear_learning_pathway(), _create_non_terminal_backprop_learning_components,
            or _create_terminal_backprop_learning_components()
          .. note::
             - also assigned LEARNING
             - must project to a LearningMechanism

        OUTPUT:
          - all TERMINAL Nodes *unless* they are:
            - a ModulatoryMechanism (i.e., ControlMechanism or LearningMechanism)
            - an ObjectiveMechanisms associated with ModulatoryMechanism
          - all Nodes that project only to:
            - a ModulatoryMechanism
            - an ObjectiveMechanism designated CONTROL_OBJECTIVE, CONTROLLER_OBJECTIVE or LEARNING_OBJECTIVE
            ? unless it is the ??TARGET_MECHANISM for a 'learning pathway <Composition_Learning_Pathway>`
              this is currently the case, but is inconsistent with the analog in Control,
              where monitored Mechanisms *are* allowed to be OUTPUT;
              therefore, might be worth allowing TARGET_MECHANISM to be assigned as OUTPUT
          - all Nodes for which OUTPUT has been assigned as a required_node_role, inclUding by user
            (i.e., in self.required_node_roles[NodeRole.OUTPUT]

        TERMINAL:
          - all Nodes that
            - are not an ObjectiveMechanism assigned the role CONTROLLER_OBJECTIVE
            - or have *no* efferent projections OR
            - or for which any efferent projections are either:
                - to output_CIM OR
                - assigned as feedback (i.e., self.graph.comp_to_vertex[efferent].feedback == EdgeType.FEEDBACK
          .. _note::
             - this insures that for cases in which there are nested CYCLES
               (e.g., LearningMechanisms for a `learning Pathway <Composition.Learning_Pathway>`),
               only the Node in the *outermost* CYCLE that is specified as a FEEDBACK_SENDER
               is assigned as a TERMINAL Node
               (i.e., the LearningMechanism responsible for the *first* `learned Projection;
               <Composition_Learning_Components>` in the `learning Pathway  <Composition.Learning_Pathway>`)
             - an ObjectiveMechanism assigned CONTROLLER_OBJECTIVE is prohibited since it and the Composition's
               `controller <Composition.controller>` are executed outside of (either before or after)
               all of the other Components of the Composition, as managed directly by the scheduler;
             - `Execution of a `Composition <Composition_Execution>` always ends with a `TERMINAL` Node,
               although some `TERMINAL` Nodes may execute earlier (i.e., if they belong to a `Pathway` that
               is shorter than the longest one in the Composition).

       """

        # Clear old roles
        self.nodes_to_roles.update({k: set() for k in self.nodes_to_roles})

        # Assign required_node_roles
        for node_role_pair in self.required_node_roles:
            self._add_node_role(node_role_pair[0], node_role_pair[1])

        # Get ORIGIN and TERMINAL Nodes using self.scheduler.consideration_queue
        if self.scheduler.consideration_queue:
            self._determine_origin_and_terminal_nodes_from_consideration_queue()

        # INPUT
        for node in self.get_nodes_by_role(NodeRole.ORIGIN):
            # Don't allow INTERNAL Nodes to be INPUTS
            if NodeRole.INTERNAL in self.get_roles_by_node(node):
                continue
            self._add_node_role(node, NodeRole.INPUT)

        # CYCLE
        for node in self.graph_processing.cycle_vertices:
            self._add_node_role(node, NodeRole.CYCLE)

        # FEEDBACK_SENDER and FEEDBACK_RECEIVER
        for receiver in self.graph_processing.vertices:
            for sender, typ in receiver.source_types.items():
                if typ is EdgeType.FEEDBACK:
                    self._add_node_role(
                        sender.component,
                        NodeRole.FEEDBACK_SENDER
                    )
                    self._add_node_role(
                        receiver.component,
                        NodeRole.FEEDBACK_RECEIVER
                    )

        # FIX 4/25/20 [JDC]:  NEED TO AVOID AUTOMATICALLY (RE-)ASSIGNING ONES REMOVED BY exclude_node_roles
        #     - Simply exclude any LEARNING_OBJECTIVE and CONTROL_OBJECTIVE that project only to ModulatoryMechanism
        #     - NOTE IN PROGRAM ERROR FAILURE TO ASSIGN CONTROL_OBJECTIVE

        # OUTPUT

        for node in self.nodes:

            # Assign OUTPUT if node is TERMINAL...
            if NodeRole.TERMINAL in self.get_roles_by_node(node):
                # unless it is a ModulatoryMechanism
                if (isinstance(node, ModulatoryMechanism_Base)
                    # # FIX: WHY WOULD SUCH AN ObjectiveMechanism BE TERMINAL IF IT PROJECTS TO A MODULATORY_MECHANISM
                    # #      (IS THIS BECAUSE MODULATORY MECH GETS DISCOUNTED FROM BEING TERMINAL IN graph_processing?)
                    # # or an ObjectiveMechanism associated with ControlMechanism or LearningMechanism
                    #     or any(role in self.get_roles_by_node(node) for role in {NodeRole.CONTROL_OBJECTIVE,
                    #                                                              NodeRole.CONTROLLER_OBJECTIVE,
                    #                                                              NodeRole.LEARNING_OBJECTIVE})
                ):
                    continue
                else:
                    self._add_node_role(node, NodeRole.OUTPUT)

            # Assign OUTPUT to any non-TERMINAL Nodes
            else:

                # IMPLEMENTATION NOTE:
                #   This version allows LEARNING_OBJECTIVE to be assigned as OUTPUT
                #   The alternate version below restricts OUTPUT only to RecurrentTransferMechasnism
                # # Assign OUTPUT if node projects only to itself and/or a LearningMechanism
                # #     (i.e., it is either a RecurrentTransferMechanism configured for learning
                # #      or the LEARNING_OBJECTIVE of a `learning pathway <Composition_Learning_Pathway>`
                # if all(p.receiver.owner is node or isinstance(p.receiver.owner, LearningMechanism)
                #        for p in node.efferents):
                #     self._add_node_role(node, NodeRole.OUTPUT)
                #     continue

                # Assign OUTPUT if it is a `RecurrentTransferMechanism` configured for learning
                #    and doesn't project to any Nodes other than its `AutoassociativeLearningMechanism`
                #    (this is not picked up as a `TERMINAL` since it projects to the `AutoassociativeLearningMechanism`)
                #    but can (or already does) project to an output_CIM
                if all((p.receiver.owner is node # <- recurrence
                        or isinstance(p.receiver.owner, AutoAssociativeLearningMechanism)
                        or p.receiver.owner is self.output_CIM) # <- already projects to an output_CIM
                       for p in node.efferents):
                    self._add_node_role(node, NodeRole.OUTPUT)
                    continue

                # Assign OUTPUT only if the node is not:
                #  - the TARGET_MECHANISM of a `learning Pathway <Composition_Learning_Pathway>`
                #  - a ModulatoryMechanism
                # and the node projects only to:
                #  - an ObjectiveMechanism designated as CONTROL_OBJECTIVE, CONTROLLER_OBJECTIVE or LEARNING_OBJECTIVE
                #  - and/or directly to a ControlMechanism but is not an ObjectiveMechanism
                #  - and/or (already projects) to output_CIM
                if NodeRole.TARGET in self.get_roles_by_node(node):
                    continue
                if isinstance(node, ModulatoryMechanism_Base):
                    continue
                if all((any(p.receiver.owner in self.get_nodes_by_role(role)
                           for role in {NodeRole.CONTROL_OBJECTIVE,
                                        NodeRole.CONTROLLER_OBJECTIVE,
                                        NodeRole.LEARNING_OBJECTIVE})
                        or p.receiver.owner is self.output_CIM
                       or (isinstance(p.receiver.owner, ControlMechanism) and not isinstance(node, ObjectiveMechanism)))
                       for p in node.efferents):
                    self._add_node_role(node, NodeRole.OUTPUT)

                # If node is a Composition and its output_CIM has OutputPorts that either have no Projections
                #     or projections to self.output_CIM, then assign as OUTPUT Node
                # Note: this ensures that if a nested Comp has both Nodes that project to ones in an outer Composition
                #       *and* legit OUTPUT Nodes, the latter qualify to make the nested Comp an OUTPUT Node
                if isinstance(node, Composition):
                    # for port in node.output_CIM.output_ports:
                    #     if (not port.efferents
                    #             or any(proj.receiver.owner is self.output_CIM for proj in port.efferents)):
                    #         self._add_node_role(node, NodeRole.OUTPUT)
                    #         break
                    if any(not port.efferents or any(proj.receiver.owner is self.output_CIM for proj in port.efferents)
                           for port in node.output_CIM.output_ports):
                        self._add_node_role(node, NodeRole.OUTPUT)

        # Assign SINGLETON and INTERNAL nodes
        for node in self.nodes:
            if all(n in self.nodes_to_roles[node] for n in {NodeRole.ORIGIN, NodeRole.TERMINAL}):
                self._add_node_role(node, NodeRole.SINGLETON)
            if not any(n in self.nodes_to_roles[node] for n in {NodeRole.ORIGIN, NodeRole.TERMINAL}):
                self._add_node_role(node, NodeRole.INTERNAL)

        # Finally, remove any NodeRole assignments specified in excluded_node_roles
        for node in self.nodes:
            for node, role in self.excluded_node_roles:
                if role in self.get_roles_by_node(node):
                    self._remove_node_role(node, role)

        # Manual override to avoid INPUT/OUTPUT setting, which would cause
        # CIMs to be created, which is not correct for controllers
        if self.controller is not None:
            self.nodes_to_roles[self.controller] = {NodeRole.CONTROLLER}

        self.needs_determine_node_roles = False

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
        try:
            self.nodes_to_roles[node].add(role)
        except KeyError:
            raise CompositionError(f"Attempt to assign {role} to '{node.name}' that is not a Node in {self.name}.")

    def _remove_node_role(self, node, role):
        if role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))
        try:
            self.nodes_to_roles[node].remove(role)
        except KeyError as e:
            pass
            # if e.args[0] is node:
            #     assert False, f"PROGRAM ERROR in _remove_node_role: {node} not found in {self.name}.nodes_to_role."
            # elif e.args[0] is role:
            #     assert False, f"PROGRAM ERROR in _remove_node_role: " \
            #                   f"{role} not found for {node} in {self.name}.nodes_to_role."
            # else:
            #     assert False, f"PROGRAM ERROR: unexpected problem in '_remove_node_role'."

    def _determine_pathway_roles(self, context=None):
        for pway in self.pathways:
            pway._assign_roles(self)

    def _get_external_modulatory_projections(self):
        """

            Returns
            -------

            list[`Modulatory Projections <ModulatoryProjection>`] :
                list of `Modulatory Projections <ModulatoryProjection>` that originate from enclosing
                `Compositions <Composition>` and that modulate a parameter of a `Node` of the current `Composition`

        """
        external_modulators = []
        for node in [i for i in self.nodes if not i.componentType == 'Composition']:
            for comp_projection in node.mod_afferents:
                sender = comp_projection.sender.owner
                receiver = comp_projection.receiver
                route_projection_through_pcim = False
                if sender not in self.nodes \
                        and not (hasattr(sender, 'composition') and sender.composition == self):
                    connections = [v for k, v in receiver._afferents_info.items()]
                    for i in connections:
                        if i.compositions:
                            for j in i.compositions:
                                if self in [v for k, v in dict(j._get_nested_nodes()).items()]:
                                    route_projection_through_pcim = True
                                    referring_composition = j
                                    external_modulators.append((comp_projection, referring_composition))
                                    break
                        if route_projection_through_pcim:
                            break
        return external_modulators

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

            - create a corresponding InputPort and ControlSignal on the `parameter_CIM <Composition.parameter_CIM>` for
              each InputPort of each node in the Composition that receives a modulatory projection from an enclosing
              Composition. Connect the original ControlSignal to the parameter_CIM's corresponding InputPort via a
              standard MappingProjection, then activate the projections that are created automatically during
              instantiation of the ControlSignals to carry that signal to the target ParameterPort.

            - build three dictionaries:

                (1) input_CIM_ports = { INPUT Node InputPort: (InputCIM InputPort, InputCIM OutputPort) }

                (2) output_CIM_ports = { OUTPUT Node OutputPort: (OutputCIM InputPort, OutputCIM OutputPort) }

                (3) parameter_CIM_ports = { ( Signal Owner, Signal Receiver ): (ParameterCIM InputPort, ParameterCIM OutputPort) }

            - if the Node has any shadows, create the appropriate projections as needed.

            - delete all of the above for any node Ports which were previously, but are no longer, classified as
              INPUT/OUTPUT
        """

        # Composition's CIMs need to be set up from scratch, so we remove their default input and output ports
        if not self.input_CIM.connected_to_composition:
            self.input_CIM.remove_ports(self.input_CIM.input_port)
            self.input_CIM.remove_ports(self.input_CIM.output_port)
            # flag the CIM as connected to the Composition so we don't remove ports on future calls to this method
            self.input_CIM.connected_to_composition = True

        if not self.output_CIM.connected_to_composition:
            self.output_CIM.remove_ports(self.output_CIM.input_port)
            self.output_CIM.remove_ports(self.output_CIM.output_port)
            # flag the CIM as connected to the Composition so we don't remove ports on future calls to this method
            self.output_CIM.connected_to_composition = True

        # PCIMs are not currently supported for compilation if they don't have any input/output ports,
        # so remove their default ports only in the case that additional ports are going to be configured below
        external_modulatory_projections = self._get_external_modulatory_projections()
        if not self.parameter_CIM.connected_to_composition and (external_modulatory_projections or len(self.parameter_CIM.input_ports) > 1):
            self.parameter_CIM.remove_ports(self.parameter_CIM.input_port)
            self.parameter_CIM.remove_ports(self.parameter_CIM.output_port)
            # flag the CIM as connected to the Composition so we don't remove ports on future calls to this method
            self.parameter_CIM.connected_to_composition = True

        # INPUT CIM
        current_input_node_input_ports = set()

        # we're going to set up ports on the input CIM for all input nodes in the Composition
        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
        for node in input_nodes:

            # loop through all external InputPorts on INPUT Nodes (i.e. Ports that are Projected to from other Nodes)
            for input_port in node.external_input_ports:

                # add it to set of current InputPorts
                current_input_node_input_ports.add(input_port)

                # if there is not a corresponding CIM InputPort/OutputPort pair, add them
                if input_port not in set(self.input_CIM_ports.keys()):
                    # instantiate the InputPort on the input CIM to correspond to the Node's InputPort
                    interface_input_port = InputPort(owner=self.input_CIM,
                                                     variable=np.atleast_2d(input_port.defaults.variable)[0],
                                                     reference_value=input_port.defaults.value,
                                                     name= INPUT_CIM_NAME + "_" + node.name + "_" + input_port.name,
                                                     context=context)

                    if NodeRole.TARGET in self.get_roles_by_node(node):
                        interface_input_port.parameters.require_projection_in_composition.set(False, override=True)

                    # add Port to the input CIM
                    self.input_CIM.add_ports([interface_input_port],
                                             context=context)

                    # instantiate the OutputPort on the input CIM to correspond to the Node's InputPort
                    interface_output_port = OutputPort(owner=self.input_CIM,
                                                       variable=(OWNER_VALUE, functools.partial(self.input_CIM.get_input_port_position, interface_input_port)),
                                                       function=Identity,
                                                       name=INPUT_CIM_NAME + "_" + node.name + "_" + input_port.name,
                                                       context=context)

                    # add Port to the input CIM
                    self.input_CIM.add_ports([interface_output_port],
                                             context=context)

                    # add entry to input_CIM_ports dict, so that the CIM ports that correspond to a given
                    # input node's InputPort can be retrieved
                    self.input_CIM_ports[input_port] = (interface_input_port, interface_output_port)

                    # create Projection from the output port on the input CIM to the input port on the input node
                    projection = MappingProjection(sender=interface_output_port,
                                                   receiver=input_port,
                                                   matrix=IDENTITY_MATRIX,
                                                   name="(" + interface_output_port.name + ") to ("
                                                        + input_port.owner.name + "-" + input_port.name + ")")

                    # activate the Projection
                    projection._activate_for_compositions(self)

                    # if the node is a nested Composition, activate the Projection for the nested Composition as well
                    if isinstance(node, Composition):
                        projection._activate_for_compositions(node)

        # compare the set of ports in input_CIM_ports to the set of input ports of input nodes that currently exist in
        # the composition, so that we can remove ports on the input CIM that correspond to nodes that no longer should
        # connect to the CIM
        sends_to_input_ports = set(self.input_CIM_ports.keys())

        # For any port still registered on the CIM that does not map to a corresponding INPUT node I.S.:
        for input_port in sends_to_input_ports.difference(current_input_node_input_ports):
            # remove the CIM input and output ports associated with this INPUT node InputPort
            self.input_CIM.remove_ports(self.input_CIM_ports[input_port][0])
            for proj in self.input_CIM_ports[input_port][1].efferents:
                self.remove_projection(proj)
            self.input_CIM.remove_ports(self.input_CIM_ports[input_port][1])
            # and from the dictionary of CIM OutputPort/InputPort pairs
            del self.input_CIM_ports[input_port]

        # OUTPUT CIM
        # loop over all OUTPUT nodes
        # Set up ports on the output CIM for all output nodes in the Composition
        current_output_node_output_ports = set()

        # loop through all output ports on OUTPUT and PROBE nodes
        for node in self.get_nodes_by_role(NodeRole.OUTPUT) + self.get_nodes_by_role(NodeRole.PROBE):
            for output_port in node.output_ports:
                current_output_node_output_ports.add(output_port)

                # if there is not a corresponding CIM InputPort/OutputPort pair, add them
                if output_port not in set(self.output_CIM_ports.keys()):

                    # instantiate the input port on the output CIM to correspond to the node's output port
                    interface_input_port = InputPort(owner=self.output_CIM,
                                                     variable=output_port.defaults.value,
                                                     reference_value=output_port.defaults.value,
                                                     name=OUTPUT_CIM_NAME + "_" + node.name + "_" + output_port.name,
                                                     context=context)

                    # add port to the output CIM
                    self.output_CIM.add_ports([interface_input_port],
                                              context=context)

                    # instantiate the OutputPort on the output CIM to correspond to the node's OutputPort
                    interface_output_port = OutputPort(
                            owner=self.output_CIM,
                            variable=(OWNER_VALUE, functools.partial(self.output_CIM.get_input_port_position,
                                                                     interface_input_port)),
                            function=Identity,
                            reference_value=output_port.defaults.value,
                            name=OUTPUT_CIM_NAME + "_" + node.name + "_" + output_port.name,
                            context=context)

                    # add port to the output CIM
                    self.output_CIM.add_ports([interface_output_port],
                                              context=context)

                    # add entry to output_CIM_ports dict, so that CIM ports that correspond to a given
                    # output node's OutputPort can be retrieved
                    self.output_CIM_ports[output_port] = (interface_input_port, interface_output_port)

                    proj_name = "(" + output_port.name + ") to (" + interface_input_port.name + ")"

                    # create Projection from the OutputPort of the output Node to InputPort on the output CIM
                    proj = MappingProjection(
                        sender=output_port,
                        receiver=interface_input_port,
                        # FIX:  This fails if OutputPorts don't all have the same dimensionality (number of axes);
                        #       see example in test_output_ports/TestOutputPorts
                        matrix=IDENTITY_MATRIX,
                        name=proj_name
                    )

                    # activate the projection
                    proj._activate_for_compositions(self)

                    # if the Node is a nested Composition, activate the Projection for the nested Composition as well
                    if isinstance(node, Composition):
                        proj._activate_for_compositions(node)

        # compare the set of ports in output_CIM_ports to the set of output ports of output nodes that currently exist
        # in the composition, so that we can remove ports on the output CIM that correspond to nodes that no longer
        # should connect to the CIM
        previous_output_node_output_ports = set(self.output_CIM_ports.keys())
        for output_port in previous_output_node_output_ports.difference(current_output_node_output_ports):
            # remove the CIM input and output ports associated with this Terminal Node OutputPort
            for proj in self.output_CIM_ports[output_port][0].path_afferents:
                self.remove_projection(proj)
            self.output_CIM.remove_ports(self.output_CIM_ports[output_port][0])
            self.output_CIM.remove_ports(self.output_CIM_ports[output_port][1])
            # and from the dictionary of CIM OutputPort/InputPort pairs
            del self.output_CIM_ports[output_port]

        # PARAMETER CIM

        # We get the projection that needs to be routed through the PCIM as well as the composition that owns it,
        # because we will need to activate the new projections for the composition that owns the PCIM as well as the
        # referring composition
        for comp_projection, referring_composition in external_modulatory_projections:
            # the port that receives the projection
            receiver = comp_projection.receiver
            # the mechanism that owns the port for which the projection is an afferent
            owner = receiver.owner
            if receiver not in self.parameter_CIM_ports:
                # control signal modulation should match the modulation type of the original control signal
                modulation = comp_projection.sender.modulation
                # input port of parameter CIM that will receive projection from the original control signal
                interface_input_port = InputPort(owner=self.parameter_CIM,
                                                 variable=receiver.defaults.value,
                                                 reference_value=receiver.defaults.value,
                                                 name= PARAMETER_CIM_NAME + "_" + owner.name + "_" + receiver.name,
                                                 # default_input=DEFAULT_VARIABLE,
                                                 context=context)
                self.parameter_CIM.add_ports([interface_input_port], context=context)
                # control signal for parameter CIM that will project directly to inner Composition's parameter
                control_signal = ControlSignal(
                        modulation=modulation,
                        variable=(OWNER_VALUE, functools.partial(self.parameter_CIM.get_input_port_position, interface_input_port)),
                        transfer_function=Identity,
                        modulates=receiver,
                        name = PARAMETER_CIM_NAME + "_"  + owner.name + "_" + receiver.name,
                )
                self.parameter_CIM.add_ports([control_signal], context=context)
                # add sender and receiver to self.parameter_CIM_ports dict
                self.parameter_CIM_ports[receiver] = (interface_input_port, control_signal)
                # projection name
                proj_name = "(" + comp_projection.sender.name + ") to (" + interface_input_port.name + ")"
                # instantiate the projection
                proj = MappingProjection(
                    sender=comp_projection.sender,
                    receiver=interface_input_port,
                    # FIX:  This fails if OutputPorts don't all have the same dimensionality (number of axes);
                    #       see example in test_output_ports/TestOutputPorts
                    matrix=IDENTITY_MATRIX,
                    name=proj_name
                )
                # activate the projection for this composition and the referring composition
                proj._activate_for_compositions(self)
                proj._activate_for_compositions(referring_composition)
                # activate all projections from the newly instantiated control signal
                for projection in control_signal.projections:
                    projection._activate_for_compositions(self)
                # remove the original direct projection from the target ParameterPort
                receiver.mod_afferents.remove(comp_projection)
                comp_projection.sender._remove_projection_from_port(comp_projection)

        for cim, type in zip([self.input_CIM, self.output_CIM, self.parameter_CIM], [INPUT, OUTPUT, PARAMETER]):

            # Enforce order of ports to same as node_order
            # Get node port mappings for cim
            node_port_to_cim_port_tuples_mapping = cim.port_map
            # Create lists of tuples of (cim_input_port, cim_output_port, index), in which indices are for
            # nodes within self.nodes (cim_node_indices) and ports within nodes (cim_port_within_node_indices
            cim_node_indices = []
            cim_port_within_node_indices = []
            for node_port, cim_ports in node_port_to_cim_port_tuples_mapping.items():
                node = node_port.owner
                if isinstance(node, CompositionInterfaceMechanism):
                    node = node.composition
                cim_node_indices.append((cim_ports[0], cim_ports[1], self.nodes.index(node)))
                node_port_list = getattr(node, f'{type}_ports')
                cim_port_within_node_indices.append((cim_ports[0], cim_ports[1], node_port_list.index(node_port)))
            # Sort cim input_ports and output_ports...
            # Note:  put any extra ports (i.e., user-assigned, despite warning!) at end of list
            #        by assigning len(self.nodes) as the default
            if node_port_to_cim_port_tuples_mapping:
                # FIX 4/28/20 [JDC]: ALSO SORT parameter_ports FOR cim??  DOES IT EVEN HAVE ANY?
                # First sort according to the order in which ports for the same Node are listed on that node
                cim.input_ports.sort(key=lambda x: next((cim_prt_tpl[2]
                                                         for cim_prt_tpl in cim_port_within_node_indices
                                                         if x in cim_prt_tpl),
                                                        len(node_port_list)))
                cim.output_ports.sort(key=lambda x: next((cim_prt_tpl[2]
                                                          for cim_prt_tpl in cim_port_within_node_indices
                                                          if x in cim_prt_tpl),
                                                         len(node_port_list)))
                # Then sort according to the order in which the Nodes appear in self.nodes
                cim.input_ports.sort(key=lambda x: next((cim_prt_tpl[2]
                                                         for cim_prt_tpl in cim_node_indices
                                                          if x in cim_prt_tpl),
                                                        len(self.nodes)))
                cim.output_ports.sort(key=lambda x: next((cim_prt_tpl[2]
                                                          for cim_prt_tpl in cim_node_indices
                                                          if x in cim_prt_tpl),
                                                         len(self.nodes)))


            # KDM 4/3/20: should reevluate this some time - is it
            # acceptable to consider _update_default_variable as
            # happening outside of this normal context? This is here as
            # a fix to the problem that when called within
            # Composition.run, context has assigned an execution_id but
            # not initialized yet. This is because _analyze_graph must
            # be called before _initialize_from_context because
            # otherwise, CIM ports will not be initialized properly
            orig_eid = context.execution_id
            context.execution_id = None
            context_string = context.string

            new_default_variable = [
                deepcopy(input_port.default_input_shape)
                for input_port in cim.input_ports
            ]

            try:
                cim._update_default_variable(new_default_variable, context)
            except MechanismError as e:

                if 'number of input_ports (0)' not in str(e):
                    raise
                # else:
                # no input ports in CIM, so assume Composition is blank

            context.execution_id = orig_eid
            context.string = context_string

            # verify there is exactly one automatically instantiated input port for each automatically instantiated
            # output port
            num_auto_input_ports = len(cim.input_ports) - len(cim.user_added_ports[INPUT_PORTS])
            num_auto_output_ports = len(cim.output_ports) - len(cim.user_added_ports[OUTPUT_PORTS])
            assert num_auto_input_ports == num_auto_output_ports
            if type==INPUT:
                # FIX 4/4/20 [JDC]: NEED TO ADD ASSERTION FOR NUMBER OF SHADOW PROJECTIONS
                n = len(cim.output_ports) - len(cim.user_added_ports[OUTPUT_PORTS])
                i = sum([len(n.external_input_ports) for n in self.get_nodes_by_role(NodeRole.INPUT)])
                assert n == i, f"PROGRAM ERROR:  Number of OutputPorts on {self.input_CIM.name} ({n}) does not match " \
                               f"the number of external_input_ports over all INPUT nodes of {self.name} ({i})."
                # p = len([p for p in self.projections if (INPUT_CIM_NAME in p.name and SHADOW_INPUT_NAME not in p.name )])
                # FIX 4/4/20 [JDC]: THIS FAILS FOR NESTED COMPS (AND OTHER PLACES?):
                # assert p == n, f"PROGRAM ERROR:  Number of Projections associated with {self.input_CIM.name})" \
                #                f"({p} does not match the number of its OutputPorts ({n})."
            elif type==OUTPUT:
                n = len(cim.input_ports) - len(cim.user_added_ports[INPUT_PORTS])
                o = sum([len(n.output_ports)
                         for n in self.get_nodes_by_role(NodeRole.PROBE) + self.get_nodes_by_role(NodeRole.OUTPUT)])
                assert n == o, f"PROGRAM ERROR:  Number of InputPorts on {self.output_CIM.name} ({n}) does not " \
                               f"match the number of OutputPorts over all OUTPUT nodes of {self.name} ({o})."
                # p = len([p for p in self.projections if OUTPUT_CIM_NAME in p.name])
                # FIX 4/4/20 [JDC]: THIS FAILS FOR NESTED COMPS (AND OTHER PLACES?):
                # assert p == n, f"PROGRAM ERROR:  Number of Projections associated with {self.output_CIM.name} " \
                #                f"({p}) does not match the number of its InputPorts ({n})."
            elif type==PARAMETER:
                # _get_external_control_projections finds all projections which currently need to be routed through the
                # PCIM, so the length of the returned array should be 0
                c = len(self._get_external_modulatory_projections())
                assert c == 0, f"PROGRAM ERROR:  Number of external control projections {c} is greater than 0. " \
                               f"This means there was a failure to route these projections through the PCIM."

    def _get_nested_node_CIM_port(self,
                                  node: Mechanism,
                                  node_port: tc.any(InputPort, OutputPort),
                                  role: tc.enum(NodeRole.INPUT, NodeRole.PROBE, NodeRole.OUTPUT)
                                  ):
        """Check for node in nested Composition
        Assign NodeRole.PROBE to relevant nodes if allow_probes is specified (see handle_probes below)
        Return relevant port of relevant CIM if found and nested Composition in which it was found; else None's
        """

        def try_assigning_as_probe(node, role, comp):
            """Try to assign node as PROBE
            If:
             - node is an INPUT or INTERNAL node in its Composition
             - outermost Composition has controller
             - allow_probes is set for it or its objective_mechanism
            Then:
             - add PROBE as one of its roles
             - call _analyze_graph() to create output_CIMs ports and projections for it
             - return True
            Else:
             - return False
            """
            err_msg = f"{node.name} found in nested {Composition.__name__} of {self.name} " \
                      f"({nc.name}) but without required {role}."

            # Get any Nodes monitored by ControlMechanisms for which allow_probes is specified
            ctl_monitored_nodes = {}
            if any(isinstance(n, ControlMechanism) and n.allow_probes for n in self._all_nodes):
                ctl_monitored_nodes = self._get_monitor_for_control_nodes()

            # If allow_probes is set on the Composition or any ControlMechanisms, then attempt to assign node as PROBE
            if self.allow_probes is True or ctl_monitored_nodes:
                # Check if Node is an INPUT or INTERNAL
                if any(role for role in comp.nodes_to_roles[node] if role in {NodeRole.INPUT, NodeRole.INTERNAL}):
                    comp._add_required_node_role(node, NodeRole.PROBE)
                    # Ignore warning since a Projection to the PROBE will not yet have been instantiated
                    # self._analyze_graph(context=Context(string='IGNORE_NO_AFFERENTS_WARNING'))
                    self._analyze_graph(context=Context(source=ContextFlags.COMPOSITION,
                                                        string='IGNORE_NO_AFFERENTS_WARNING'))
                    return

            # Failed to assign node as PROBE, so get ControlMechanisms that may be trying to monitor it
            ctl_monitored_nodes = self._get_monitor_for_control_nodes()
            if node in ctl_monitored_nodes:
                if ctl_monitored_nodes[node].objective_mechanism:
                    # Node was specified for monitoring by an ObjectiveMechanism of a ControlMechanism
                    raise CompositionError(err_msg + f" Try setting '{ALLOW_PROBES}' argument of ObjectiveMechanism "
                                                     f"for {ctl_monitored_nodes[node].name} to 'True'.")
                # Node was specified for monitoring by ControlMechanism
                raise CompositionError(err_msg + f" Try setting '{ALLOW_PROBES}' argument "
                                                 f"of {ctl_monitored_nodes[node].name} to 'True'.")
            # Node was not specified for monitoring by a ControlMechanism
            raise CompositionError(err_msg)

        nested_comp = CIM_port_for_nested_node = CIM = None

        nested_comps = [i for i in self.nodes if isinstance(i, Composition)]
        for nc in nested_comps:
            nested_nodes = dict(nc._get_nested_nodes())
            if node in nested_nodes or node in nc.nodes.data:
                owning_composition = nc if node in nc.nodes else nested_nodes[node]
                # Must be assigned Node.Role of INPUT, PROBE, or OUTPUT (depending on receiver vs sender)
                # This validation does not apply to ParameterPorts. Externally modulated nodes
                # can be in any position within a Composition. They don't need to be INPUT or OUTPUT nodes.
                if not isinstance(node_port, ParameterPort) and role not in owning_composition.nodes_to_roles[node]:
                    try_assigning_as_probe(node, role, owning_composition)
                # With the current implementation, there should never be multiple nested compositions that contain the
                # same mechanism -- because all nested compositions are passed the same execution ID
                # FIX: 11/15/21:  ??WHY IS THIS COMMENTED OUT:
                # if CIM_port_for_nested_node:
                #     warnings.warn("{} found with {} of {} in more than one nested {} of {}; "
                #                   "only first one found (in {}) will be used".
                #                   format(node.name, NodeRole.__name__, repr(role),
                #                          Composition.__name__, self.name, nested_comp.name))
                #     continue
                if isinstance(node_port, InputPort):
                    if node_port in nc.input_CIM_ports:
                        CIM_port_for_nested_node = owning_composition.input_CIM_ports[node_port][0]
                        CIM = owning_composition.input_CIM
                    else:
                        nested_node_CIM_port_spec = nc._get_nested_node_CIM_port(node, node_port, NodeRole.INPUT)
                        CIM_port_for_nested_node = nc.input_CIM_ports[nested_node_CIM_port_spec[0]][0]
                        CIM = nc.input_CIM
                elif isinstance(node_port, OutputPort):
                    if node_port in nc.output_CIM_ports:
                        CIM_port_for_nested_node = owning_composition.output_CIM_ports[node_port][1]
                        CIM = owning_composition.output_CIM
                    else:
                        nested_node_CIM_port_spec = nc._get_nested_node_CIM_port(node,
                                                                                 node_port,
                                                                                 role)
                                                                                 # NodeRole.OUTPUT)
                        CIM_port_for_nested_node = nc.output_CIM_ports[nested_node_CIM_port_spec[0]][1]
                        CIM = nc.output_CIM
                elif isinstance(node_port, ParameterPort):
                    # NOTE: there is special casing here for parameter ports. They don't have a node role
                    # associated with them in the way that input and output nodes do, so we don't know for sure
                    # if they will have a port in parameter_CIM_ports. If they don't, we just set the
                    # CIM_port_for_nested_node to the node_port itself, and delegate its routing through the PCIM
                    # to a future call to create_CIM_ports
                    # if node_port in nc.parameter_CIM_ports:
                    #     CIM_port_for_nested_node = nc.parameter_CIM_ports[node_port][0]
                    #     CIM = nc.parameter_CIM
                    # else:
                    CIM_port_for_nested_node = node_port
                    CIM = nc.parameter_CIM
                nested_comp = nc
                break

        # Return CIM_port_for_nested_node in both expected node and node_port slots
        return CIM_port_for_nested_node, CIM_port_for_nested_node, nested_comp, CIM

    # endregion NODES

    # ******************************************************************************************************************
    # region ----------------------------------- PROJECTIONS -----------------------------------------------------------
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

    @handle_external_context(source=ContextFlags.METHOD)
    def add_projection(self,
                       projection=None,
                       sender=None,
                       receiver=None,
                       feedback=False,
                       learning_projection=False,
                       name=None,
                       allow_duplicates=False,
                       context=None
                       ):
        """Add **projection** to the Composition.

        If **projection** is not specified, create a default `MappingProjection` using **sender** and **receiver**.

        If **projection** is specified:

        • if **projection** has already been instantiated, and **sender** and **receiver** are also specified,
          they must match the `sender <MappingProjection.sender>` and `receiver <MappingProjection.receiver>`
          of **projection**.

        • if **sender** and **receiver** are specified and one or more Projections already exists between them:
          - if it is in the Composition:
            - if there is only one, the request is ignored and the existing Projection is returned
            - if there is more than one, an exception is raised as this should never be the case
          - if it is NOT in the Composition:
            - if there is only one, that Projection is used;
            - if there is more than one, the last in the list (presumably the most recent) is used;
            in either case, processing continues, to activate it for the Composition,
            construct any "shadow" projections that may be specified, and assign feedback if specified.

        • if the status of **projection** is `deferred_init`:

          - if its `sender <Projection_Base.sender>` and/or `receiver <Projection_Base.receiver>` attributes are not
            specified, then **sender** and/or **receiver** are used.

          - if `sender <Projection_Base.sender>` and/or `receiver <Projection_Base.receiver>` attributes are specified,
            they must match **sender** and/or **receiver** if those have also been specified.

          - if a Projection between the specified sender and receiver does *not* already exist, it is initialized; if
            it *does* already exist, the request to add it is ignored, however requests to shadow it and/or mark it as
            a `feedback` Projection are implemented (in case it has not already been done for the existing Projection).

        .. note::
           If **projection** is an instantiated Projection (i.e., not in `deferred_init`), and one already exists
           between its `sender <Projection_Base.sender>` and `receiver <Projection_Base.receiver>`, a warning is
           generated and the request is ignored.

        .. technical_note::
            Duplicates are determined by the `Ports <Port>` to which they project, not the `Mechanisms <Mechanism>`
            (to allow multiple Projections to exist between the same pair of Mechanisms using different Ports).
            ..
            If an already instantiated Projection is passed to add_projection and is a duplicate of an existing one,
            it is detected and suppressed, with a warning, in Port._instantiate_projections_to_port.
            ..
            If a Projection with deferred_init status is a duplicate, it is fully suppressed here,
            as these are generated by add_linear_processing_pathway if the pathway overlaps with an existing one,
            and so warnings are unnecessary and would be confusing to users.

        Arguments
        ---------

        sender : Mechanism, Composition, or OutputPort
            the sender of **projection**.

        projection : Projection, matrix
            the projection to add.

        receiver : Mechanism, Composition, or InputPort
            the receiver of **projection**.

        feedback : bool or FEEDBACK : False
            if False, the Projection is *never* designated as a `feedback Projection
            <Composition_Feedback_Designation>`, even if that may have been the default behavior (e.g.,
            for a `ControlProjection` that forms a `loop <Composition_Cycle_Structure>`; if True or *FEEDBACK*,
            and the Projection is in a loop, it is *always* designated as a feedback Projection, and used to `break"
            the cycle <Composition_Feedback_Designation>`.

        Returns
        -------

        `Projection` :
            `Projection` if added, else None

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
                # If either the sender or receiver are not in Composition and are not CompositionInterfaceMechanisms
                #   remove the Projection and its inclusion in any relevant Port attributes
                if ((not isinstance(sender_check, CompositionInterfaceMechanism)
                     and sender_check not in self.nodes)
                        or (not isinstance(receiver_check, CompositionInterfaceMechanism)
                            and receiver_check not in self.nodes)):
                    for proj in existing_projections:
                        self.remove_projection(proj)
                        for port in sender_check.output_ports + receiver_check.input_ports:
                            port.remove_projection(proj, context=context)
                else:
                    #  Need to do stuff at end, so can't just return
                    if self.prefs.verbosePref:
                        warnings.warn(f"Several existing projections were identified between "
                                      f"{sender.name} and {receiver.name}: {[p.name for p in existing_projections]}; "
                                      f"the last of these will be used in {self.name}.")
                    projection = existing_projections[-1]

        # If Projection is one that is instantiated and is directly between Nodes in nested Compositions,
        #   then re-specify it so that the proper routing can be instantiated between those Compositions
        # Note: restrict to PathwayProjections, since routing of ModulatoryProjections is handled separately.
        elif (isinstance(projection, PathwayProjection_Base)
              and projection._initialization_status is ContextFlags.INITIALIZED):
            sender_node = projection.sender.owner
            receiver_node = projection.receiver.owner
            # If sender or receiver is in a nested Node
            if ((sender_node not in self.nodes
                 and sender_node in [n[0] for n in self._get_nested_nodes()])
                    or (receiver_node not in self.nodes
                         and receiver_node in [n[0] for n in self._get_nested_nodes()])):
                proj_spec = {PROJECTION_TYPE:projection.className,
                              PROJECTION_PARAMS:{
                                  FUNCTION:projection.function,
                                  MATRIX:projection.matrix.base}
                              }
                return self.add_projection(proj_spec, sender=projection.sender, receiver=projection.receiver)

        # Create Projection if it doesn't exist
        try:
            # Note: this does NOT initialize the Projection if it is in deferred_init
            projection = self._instantiate_projection_from_spec(projection, name)
        except DuplicateProjectionError:
            # return projection
            return

        # Parse sender and receiver specs
        sender, sender_mechanism, graph_sender, nested_compositions = self._parse_sender_spec(projection, sender)
        receiver, receiver_mechanism, graph_receiver, receiver_input_port, nested_compositions, learning_projection = \
            self._parse_receiver_spec(projection, receiver, sender, learning_projection)

        if (isinstance(receiver_mechanism, (CompositionInterfaceMechanism))
                and receiver_input_port.owner not in self.nodes
                and receiver.componentType == 'ParameterPort'):
            # unlike when projecting to nested InputPorts, we don't know for sure whether
            # intermediary pcims will have input ports that correspond to the ParameterPorts we are interested
            # in projecting to. the below method handles routing through intermediary pcims, including by adding needed
            # ports and projections when they don't already exist, and returns a modified projection spec for us to use
            # on this level of nesting.
            if isinstance(receiver, ParameterPort):
                projection = self._route_control_projection_through_intermediary_pcims(projection, sender,
                                                                                       sender_mechanism, receiver,
                                                                                       graph_receiver, context)
                receiver = projection.receiver

        if sender_mechanism is self.parameter_CIM:
            idx = self.parameter_CIM.output_ports.index(sender)
            in_p = self.parameter_CIM.input_ports[idx]
            out_p = self.parameter_CIM.output_ports[idx]
            self.parameter_CIM.port_map[receiver] = (in_p, out_p)

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
                projection._init_args[SENDER] = sender
                projection._init_args[RECEIVER] = receiver
                projection._deferred_init()

        else:
            existing_projections = self._check_for_existing_projections(projection, sender=sender, receiver=receiver)

        # # FIX: JDC HACK 6/13/19 to deal with projection from user-specified INPUT node added to the Composition
        # #      that projects directly to the Target node of a nested Composition
        # # If receiver_mechanism is a Target Node in a nested Composition
        # if any((n is receiver_mechanism and receiver_mechanism in nested_comp.get_nodes_by_role(NodeRole.TARGET))
        #        for nested_comp in self.nodes if isinstance(nested_comp, Composition) for n in nested_comp.nodes):
        #     # cim_target_input_port = receiver_mechanism.afferents[0].sender.owner.port_map[receiver.input_port][0]
        #     cim = next((proj.sender.owner for proj in receiver_mechanism.afferents
        #                 if isinstance(proj.sender.owner, CompositionInterfaceMechanism)), None)
        #     assert cim, f'PROGRAM ERROR: Target mechanisms {receiver_mechanism.name} ' \
        #                 f'does not receive projection from input_CIM'
        #     cim_target_input_port = cim.port_map[receiver.input_port][0]
        #     projection.receiver._remove_projection_to_port(projection)
        #     # self.remove_projection(projection)
        #     projection = MappingProjection(sender=sender, receiver=cim_target_input_port)
        #     receiver_mechanism = cim
        #     receiver = cim_target_input_port

        # FIX: KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to Composition as processing components
        if (sender_mechanism != self.input_CIM
                and sender_mechanism != self.parameter_CIM
                and sender_mechanism != self.controller
                and receiver_mechanism != self.output_CIM
                and receiver_mechanism != self.controller
                and projection not in [vertex.component for vertex in self.graph.vertices]
                and not learning_projection):

            projection.is_processing = False
            # KDM 5/24/19: removing below rename because it results in several existing_projections
            # projection.name = f'{sender} to {receiver}'

            # check for required role specification of feedback projections
            for node, role in self.required_node_roles:
                if (
                    (node == projection.sender.owner and role == NodeRole.FEEDBACK_SENDER)
                    or (node == projection.receiver.owner and role == NodeRole.FEEDBACK_RECEIVER)
                ):
                    feedback = True

            self.graph.add_component(projection, feedback=feedback)

            try:
                self.graph.connect_components(graph_sender, projection)
                self.graph.connect_components(projection, graph_receiver)
            except CompositionError as c:
                raise CompositionError(f"{c.args[0]} to {self.name}.")

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
        #   as these conditions should apply to the exisiting one (and it won't hurt to try again if they do)

        # if feedback in {True, FEEDBACK}:
        #     self.feedback_senders.add(sender_mechanism)
        #     self.feedback_receivers.add(receiver_mechanism)

        self.needs_determine_node_roles = True
        return projection

    def _add_projection(self, projection):
        self.projections.append(projection)

    def remove_projection(self, projection):
        # step 1 - remove Vertex from Graph
        if projection in [vertex.component for vertex in self.graph.vertices]:
            vert = self.graph.comp_to_vertex[projection]
            self.graph.remove_vertex(vert)
        # step 2 - remove Projection from Composition's list
        if projection in self.projections:
            self.projections.remove(projection)

        # step 3 - deactivate Projection in this Composition
        projection._deactivate_for_compositions(self)

        # step 4 - deactivate any learning to this Projection
        for param_port in projection.parameter_ports:
            for proj in param_port.mod_afferents:
                self.remove_projection(proj)
                if isinstance(proj.sender.owner, LearningMechanism):
                    for path in self.pathways:
                        # TODO: make learning_components values consistent type
                        try:
                            learning_mechs = path.learning_components['LEARNING_MECHANISMS']
                        except KeyError:
                            continue

                        if isinstance(learning_mechs, LearningMechanism):
                            learning_mechs = [learning_mechs]

                        if proj.sender.owner in learning_mechs:
                            for mech in learning_mechs:
                                self.remove_node(mech)
                            self.remove_node(path.learning_components['objective_mechanism'])
                            self.remove_node(path.learning_components['TARGET_MECHANISM'])

        # step 5 - TBI? remove Projection from afferents & efferents lists of any node


    def _validate_projection(self,
                             projection,
                             sender, receiver,
                             graph_sender,
                             graph_receiver,
                             learning_projection,
                             ):

        # FIX: [JDC 6/8/19] SHOULDN'T THERE BE A CHECK FOR THEM IN LearningProjections? OR ARE THOSE DONE ELSEWHERE?
        # Skip this validation on learning projections because they have non-standard senders and receivers
        if not learning_projection:
            if projection.sender.owner != graph_sender:
                raise CompositionError(f"Sender ('{sender.name}') assigned to '{projection.name} is "
                                       f"incompatible with the positions of these Components in '{self.name}'.")
            if projection.receiver.owner != graph_receiver:
                raise CompositionError(f"Receiver ('{receiver.name}') assigned to '{projection.name} is "
                                       f"incompatible with the positions of these Components in '{self.name}'.")

    def _instantiate_projection_from_spec(self, projection, sender=None, receiver=None, name=None):
        if isinstance(projection, dict):
            proj_type = projection.pop(PROJECTION_TYPE, None) or MappingProjection
            params = projection.pop(PROJECTION_PARAMS, None)
            projection = MappingProjection(params=params)
        elif isinstance(projection, (np.ndarray, np.matrix, list)):
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
                and sender_mechanism not in self.nodes
                and sender_mechanism != self.controller):
            if isinstance(sender, Port):
                sender_name = sender.full_name
            else:
                sender_name = sender.name

            # if the sender is in a nested Composition AND sender is an OUTPUT Node
            # then use the corresponding CIM on the nested comp as the sender going forward
            # (note:  NodeRole.OUTPUT used even for PROBES, since those currently use same output_CIMS as OUTPUT nodes)
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
            if (projection.sender.owner != sender
                    and projection.sender.owner != graph_sender
                    and projection.sender.owner != sender_mechanism):
                raise CompositionError(f"The position of {projection.name} in {self.name} "
                                       f"conflicts with its sender ({sender.name}).")

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
                and receiver_mechanism != self.controller
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

    def _update_shadow_projections(self, context=None):
        """Instantiate any missing shadow_projections that have been specified in Composition
        """

        # FIX 12/2/21: RENAME input_port -> shadowing_input_port
        def _instantiate_missing_shadow_projections(input_port, projections):
            """Instantiate shadow Projections that don't yet exist.

            **input_port** is InputPort to receive shadow Projections
            **projections** are Projections to be shadowed

            Search recursively (i.e., including in nested Compositions) for receiver(s) of projections.
            Instantiate any shadow Projections for them that don't yet exist.
            Return actual senders of all shadow Projections.
            """

            def _get_correct_sender(comp, shadowed_projection):
                """Search down the hierarchy of nested Compositions for Projection to shadow"""
                if shadowed_projection in comp.projections:
                    return shadowed_projection.sender
                else:
                    # Search for sender in INPUT Nodes of nested Compositions that are themselves INPUT Nodes
                    nested_input_comps = [nested_comp for nested_comp in comp._get_nested_compositions()
                                    if nested_comp in comp.get_nodes_by_role(NodeRole.INPUT)]
                    for comp in nested_input_comps:
                        if shadowed_projection in comp.projections:
                            return _get_sender_at_right_level(shadowed_projection)
                        else:
                            return _get_correct_sender(comp, shadowed_projection)
                    return None

            def _get_sender_at_right_level(shadowed_proj):
                """Search back up hierarchy of nested Compositions for sender at same level as **input_port**"""
                if not isinstance(shadowed_proj.sender.owner, CompositionInterfaceMechanism):
                    raise CompositionError(f"Attempt to shadow the input to a node "
                                           f"({shadowed_proj.receiver.owner.name}) in a nested Composition "
                                           f"of {self.name} that is not an INPUT Node of that Composition is "
                                           f"not currently supported.")
                else:
                    #                                    WANT THIS ONE'S SENDER
                    #                       item[0]           item[1,0]            item[1,1]
                    #  CIM MAP ENTRIES:  [SHADOWED PORT,  [input_CIM InputPort,  input_CIM OutputPort]]
                    sender_proj = [entry[1][0]
                                   for entry in list(shadowed_proj.sender.owner.port_map.items())
                                   if entry[1][1] is shadowed_proj.sender][0].path_afferents[0]
                    if input_port.owner in sender_proj.sender.owner.composition._all_nodes:
                        return sender_proj.sender
                    else:
                        return _get_sender_at_right_level(sender_proj)

            original_senders = set()
            for shadowed_projection in projections:
                correct_sender = _get_correct_sender(self, shadowed_projection)
                if correct_sender:
                    original_senders.add(correct_sender)
                    shadow_found = False
                    # Look for existing shadow_projections from correct_sender to shadowing input_port
                    for shadow_projection in input_port.path_afferents:
                        if shadow_projection.sender == correct_sender:
                            shadow_found = True
                            break
                    if not shadow_found:
                        # TBI - Shadow projection type? Matrix value?
                        new_projection = MappingProjection(sender=correct_sender,
                                                           receiver=input_port)
                        self.add_projection(new_projection, sender=correct_sender, receiver=input_port)
                else:
                    raise CompositionError(f"Unable to find port specified to be shadowed by '{input_port.owner.name}' "
                                           f"({shadowed_projection.receiver.owner.name}"
                                           f"[{shadowed_projection.receiver.name}]) within the same Composition "
                                           f"('{self.name}'), nor in any nested within it. "
                                           f"'{shadowed_projection.receiver.owner.name}' may be in another "
                                           f"Composition at the same level within '{self.name}' or in an outer "
                                           f"Composition, neither of which are supported by shadowing.")
            return original_senders

        if self.shadowing_dict.items:
            for shadowing_port, shadowed_port in self.shadowing_dict.items():
                senders = _instantiate_missing_shadow_projections(shadowing_port,
                                                                  shadowed_port.path_afferents)
                for shadow_projection in shadowing_port.path_afferents:
                    if shadow_projection.sender not in senders:
                        self.remove_projection(shadow_projection)
                        Projection_Base._delete_projection(shadow_projection)
                        if not shadow_projection.sender.efferents:
                            if isinstance(shadow_projection.sender.owner, CompositionInterfaceMechanism):
                                ports = shadow_projection.sender.owner.port_map.pop(shadow_projection.receiver)
                                shadow_projection.sender.owner.remove_ports(list(ports))
                            else:
                                shadow_projection.sender.owner.remove_ports(shadow_projection.sender)
            self._determine_node_roles(context=context)

    def _check_for_projection_assignments(self, context=None):
        """Check that all Projections and Ports with require_projection_in_composition attribute are configured.
        Validate that all InputPorts with require_projection_in_composition == True have an afferent Projection.
        Validate that all OutputPorts with require_projection_in_composition == True have an efferent Projection.
        Validate that all Projections have senders and receivers.
        Issue warning if any Projections are to/from nodes not in Composition.projections
        """
        projections = self.projections.copy()

        for node in self.nodes:
            if isinstance(node, Projection):
                projections.append(node)
                continue

            if context.source != ContextFlags.INITIALIZING and context.string != 'IGNORE_NO_AFFERENTS_WARNING':
                for input_port in node.input_ports:
                    if input_port.require_projection_in_composition \
                            and not input_port.path_afferents and not input_port.default_input:
                        warnings.warn(f"{InputPort.__name__} ('{input_port.name}') of '{node.name}' "
                                      f"doesn't have any afferent {Projection.__name__}s.")
                for output_port in node.output_ports:
                    if output_port.require_projection_in_composition and not output_port.efferents:
                        warnings.warn(f"{OutputPort.__name__} ('{output_port.name}') of '{node.name}' "
                                      f"doesn't have any efferent {Projection.__name__}s in '{self.name}'.")

        for projection in projections:
            if not projection.sender:
                warnings.warn(f'{Projection.__name__} {projection.name} is missing a sender.')
            if not projection.receiver:
                warnings.warn(f'{Projection.__name__} {projection.name} is missing a receiver.')

    def _check_for_unused_projections(self, context):
        """Warn if there are any Nodes in the Composition, or any nested within it, that are not used.
        """
        unused_projections = []
        for node in self.nodes:
            if isinstance(node, Composition):
                node._check_for_unused_projections(context)
            if isinstance(node, Mechanism):
                for proj in [p for p in node.projections if p not in self.projections]:
                    proj_deferred = proj._initialization_status & ContextFlags.DEFERRED_INIT
                    proj_name = proj._name if proj_deferred else proj.name
                    if proj in node.afferents:
                        first_item = '' if proj_deferred else f" (to '{node.name}'"
                        second_item = '' if proj_deferred else f" from '{proj.sender.owner.name}')."
                    if proj in node.efferents:
                        first_item = '' if proj_deferred else f" (from '{node.name}'"
                        second_item = '' if proj_deferred else f" to '{proj.receiver.owner.name}')."
                    unused_projections.append(f"{proj_name}{first_item}{second_item}")
        if unused_projections:
            warning = f"\nThe following Projections were specified but are not being used by Nodes in '{self.name}':"
            warnings.warn(warning + "\n\t" + "\n\t".join(unused_projections))
        self._need_check_for_unused_projections = False

    def get_feedback_status(self, projection):
        """Return True if **projection** is designated as a `feedback Projection <Composition_Feedback_Designation>`
        in the Composition, else False.
        """
        return projection in self.feedback_projections

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
            try:
                if isinstance(sender, Mechanism):
                    err_msg = f"'{sender.name}' does not have an '{OutputPort.__name__}'."
                    sender = sender.output_port
                elif isinstance(sender, Composition):
                    if not sender.nodes:
                        err_msg = f"'{self.name}' does not have any nodes that can project to '{receiver.name}'."
                        raise IndexError
                    sender = sender.output_CIM.output_port
                if isinstance(receiver, Mechanism):
                    err_msg = f"'{receiver.name}' does not have an {InputPort.__name__}."
                    receiver = receiver.input_port
                elif isinstance(receiver, Composition):
                    if not receiver.nodes:
                        err_msg = f'{self.name} does not have any nodes that can project to {receiver.name}.'
                        raise IndexError
                    receiver = receiver.input_CIM.input_port
            except IndexError:
                err_msg = f"Can't create a {Projection.__name__} from '{sender.name}' to '{receiver.name}': " + err_msg
                raise CompositionError(err_msg)

        # Check for existing Projections from specified sender
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

    def _check_for_unnecessary_feedback_projections(self):
        """
            Warn if there exist projections in the graph that the user
            labeled as EdgeType.FEEDBACK (True) but are not in a cycle
        """
        unnecessary_feedback_specs = []
        cycles = self.graph.get_strongly_connected_components()

        for proj in self.projections:
            try:
                vert = self.graph.comp_to_vertex[proj]
                if vert.feedback is EdgeType.FEEDBACK:
                    for c in cycles:
                        if proj in c:
                            break
                    else:
                        unnecessary_feedback_specs.append(proj)
            except KeyError:
                pass

        if unnecessary_feedback_specs:
            warnings.warn(
                'The following projections were labeled as feedback, '
                'but they are not in any cycles: {0}'.format(
                    ', '.join([str(x) for x in unnecessary_feedback_specs])
                )
            )

    def _check_for_nesting_with_absolute_conditions(self, scheduler, termination_conds=None):
        if any(isinstance(n, Composition) for n in self.nodes):
            interval_conds = set()
            fixed_point_conds = set()
            for _, cond in scheduler.get_absolute_conditions(termination_conds).items():
                if len(cond.absolute_intervals) > 0:
                    interval_conds.add(cond)
                if scheduler.mode == SchedulingMode.EXACT_TIME:
                    if len(cond.absolute_fixed_points) > 0:
                        fixed_point_conds.add(cond)

            warn_str = f'{self} contains a nested Composition, which may cause unexpected behavior ' \
                       f'in absolute time conditions or failure to terminate execution.'
            warn = False
            if len(interval_conds) > 0:
                warn_str += '\nFor repeating intervals:\n\t'
                warn_str += '\n\t'.join([f'{cond.owner}: {cond}\n\t\tintervals: {cond.absolute_intervals}'
                                         for cond in interval_conds])
                warn = True
            if len(fixed_point_conds) > 0:
                warn_str += '\nIn EXACT_TIME SchedulingMode, strict time points:\n\t'
                warn_str += '\n\t'.join([f'{cond.owner}: {cond}\n\t\tstrict time points: {cond.absolute_fixed_points}'
                                         for cond in fixed_point_conds])
                warn = True

            if warn:
                warnings.warn(warn_str)

    def _get_source(self, projection):
        """Return tuple with port, node and comp of sender for **projection** (possibly in a nested Composition)."""
        # Note:  if Projection is shadowing the input to a Node, the information returned will be for
        #        the output_port of the input_CIM that projects to the shadowed Node.
        port = projection.sender
        if port.owner in self.nodes:
            return (port, port.owner, self)
        elif isinstance(port.owner, CompositionInterfaceMechanism):
            return port.owner._get_source_info_from_output_CIM(port)
        else:
            # Get info for nested node
            node, comp = next((item for item in self._get_nested_nodes() if item[0] is port.owner), (None, None))
            if node:
                return (port, node, comp)
            else:
                raise CompositionError(f"No source found for {projection.name} in {self.name}.")

    def _get_destination(self, projection):
        """Return tuple with port, node and comp of receiver for **projection** (possibly in a nested Composition)."""
        port = projection.receiver
        if isinstance(port.owner, CompositionInterfaceMechanism):
            if isinstance(projection.sender.owner, ModulatoryMechanism_Base):
                return port.owner._get_modulated_info_from_parameter_CIM(port)
            else:
                return port.owner._get_destination_info_from_input_CIM(port)
        else:
            return (port, port.owner, self)

    # endregion PROJECTIONS

    # ******************************************************************************************************************
    # region ------------------------------------- PATHWAYS ------------------------------------------------------------
    # ******************************************************************************************************************

    # region ----------------------------------  PROCESSING  -----------------------------------------------------------

    def _parse_pathway(self, pathway, name, pathway_arg_str):
        from psyneulink.core.compositions.pathway import Pathway, _is_pathway_entry_spec

        # Deal with Pathway() or tuple specifications
        if isinstance(pathway, Pathway):
            # Give precedence to name specified in call to add_linear_processing_pathway
            pathway_name = name or pathway.name
            pathway = pathway.pathway
        else:
            pathway_name = name

        if isinstance(pathway, tuple):
            # If tuple is just a single Node specification for a pathway, return in list:
            if _is_pathway_entry_spec(pathway, NODE):
                pathway = [pathway]
            # If tuple is used to specify a sequence of nodes, convert to list (even though not documented):
            elif all(_is_pathway_entry_spec(n, ANY) for n in pathway):
                pathway = list(pathway)
            # If tuple is (pathway, LearningFunction), get pathway and ignore LearningFunction
            elif isinstance(pathway[1],type) and issubclass(pathway[1], LearningFunction):
                warnings.warn(f"{LearningFunction.__name__} found in specification of {pathway_arg_str}: {pathway[1]}; "
                              f"it will be ignored")
                pathway = pathway[0]
            else:
                raise CompositionError(f"Unrecognized tuple specification in {pathway_arg_str}: {pathway}")
        elif not isinstance(pathway, collections.abc.Iterable) or all(_is_pathway_entry_spec(n, ANY) for n in pathway):
            pathway = convert_to_list(pathway)
        else:
            bad_entry_error_msg = f"The following entries in a pathway specified for '{self.name}' are not " \
                                  f"a Node (Mechanism or Composition) or a Projection nor a set of either: "
            bad_entries = [repr(entry) for entry in pathway if not _is_pathway_entry_spec(entry, ANY)]
            raise CompositionError(f"{bad_entry_error_msg}{','.join(bad_entries)}")
            # raise CompositionError(f"Unrecognized specification in {pathway_arg_str}: {pathway}")

        lists = [entry for entry in pathway
                 if isinstance(entry, list) and all(_is_pathway_entry_spec(node, NODE) for node in entry)]
        if lists:
            raise CompositionError(f"Pathway specification for {pathway_arg_str} has embedded list(s): {lists}")
        return pathway, pathway_name

    # FIX: REFACTOR TO TAKE Pathway OBJECT AS ARGUMENT
    def add_pathway(self, pathway):
        """Add an existing `Pathway <Composition_Pathways>` to the Composition

        Arguments
        ---------

        pathway : the `Pathway <Composition_Pathways>` to be added

        """

        # identify nodes and projections
        nodes, projections = [], []
        for c in pathway.graph.vertices:
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

    @handle_external_context()
    def add_pathways(self, pathways, context=None):
        """Add pathways to the Composition.

        Arguments
        ---------

        pathways : Pathway or list[Pathway]
            specifies one or more `Pathways <Pathway>` to add to the Composition.  Any valid form of `Pathway
            specification <Pathway_Specification>` can be used.  A set can also be used, all elements of which are
            `Nodes <Composition_Nodes>`, in which case a separate `Pathway` is constructed for each.

        Returns
        -------

        list[Pathway] :
            List of `Pathways <Pathway>` added to the Composition.

        """

        # Possible specifications for **pathways** arg:
        #  Node specs (single or set):
        #  0  Single node:  NODE
        #  1  Set:  {NODE...} -> generate a Pathway for each NODE
        #  Single pathway spec (list, tuple or dict):
        #  2   single list:   PWAY = [NODE] or [NODE...] in which *all* are NODES with optional intercolated Projections
        #  2.5 single with sets: PWAY = [NODE or {NODE...}] or [NODE or {NODE...}, NODE or {NODE...}...]
        #  3   single tuple:  (PWAY, LearningFunction) = (NODE, LearningFunction) or
        #                                                 ([NODE...], LearningFunction)
        #  4   single dict:   {NAME: PWAY} = {NAME: NODE} or
        #                                    {NAME: [NODE...]} or
        #                                    {NAME: ([NODE...], LearningFunction)}
        #  Multiple pathway specs (in outer list):
        #  5   list with list(s): [PWAY] = [NODE, [NODE]] or [[NODE...]...]
        #  6   list with tuple(s):  [(PWAY, LearningFunction)...] = [(NODE..., LearningFunction)...] or
        #                                                       [([NODE...], LearningFunction)...]
        #  7   list with dict: [{NAME: PWAY}...] = [{NAME: NODE...}...] or
        #                                          [{NAME: [NODE...]}...] or
        #                                          [{NAME: (NODE, LearningFunction)}...] or
        #                                          [{NAME: ([NODE...], LearningFunction)}...]

        from psyneulink.core.compositions.pathway import Pathway, _is_node_spec, _is_pathway_entry_spec

        if context.source == ContextFlags.COMMAND_LINE:
            pathways_arg_str = f"'pathways' arg for the add_pathways method of {self.name}"
        elif context.source == ContextFlags.CONSTRUCTOR:
            pathways_arg_str = f"'pathways' arg of the constructor for {self.name}"
        else:
            assert False, f"PROGRAM ERROR:  unrecognized context passed to add_pathways of {self.name}."
        context.string = pathways_arg_str

        if not pathways:
            return

        # Possibilities 0, 3 or 4 (single NODE, set of NODESs tuple, dict or Pathway specified, so convert to list
        if _is_node_spec(pathways) or isinstance(pathways, (tuple, dict, Pathway)):
            pathways = convert_to_list(pathways)

        # Possibility 1 (set of Nodes): create a Pathway for each Node (since set is in pathways arg)
        elif isinstance(pathways, set):
            pathways = [pathways]

        # Possibility 2 (list is a single pathway spec) or 2.5 (includes one or more sets):
        if (isinstance(pathways, list) and
                # First item must be a node_spec or set of them
                ((_is_node_spec(pathways[0])
                  or (isinstance(pathways[0], set) and all(_is_node_spec(item) for item in pathways[0])))
                # All other items must be either Nodes, Projections or sets
                 and all(_is_pathway_entry_spec(p, ANY) for p in pathways))):
            # Place in outter list (to conform to processing of multiple pathways below)
            pathways = [pathways]
            # assert False, f"GOT TO POSSIBILITY 2" # SHOULD HAVE BEEN DONE ABOVE

        # If pathways is not now a list it must be illegitimate
        if not isinstance(pathways, list):
            raise CompositionError(f"The {pathways_arg_str} must be a "
                                   f"Node, list, set, tuple, dict or Pathway object: {pathways}.")

        # pathways should now be a list in which each entry should be *some* form of pathway specification
        #    (including original spec as possibilities 5, 6, or 7)

        # If there are any lists of Nodes in pathway, or a Pathway or dict with such a list,
        #     then treat ALL entries as parallel pathways, and embed in lists"
        if (isinstance(pathways, collections.abc.Iterable)
                and any(isinstance(pathway, (list, dict, Pathway))) for pathway in pathways):
            pathways = [pathway if isinstance(pathway, (list, dict, Pathway)) else [pathway] for pathway in pathways]
        else:
            # Put single pathway in outer list for consistency of handling below (with specified pathway as pathways[0])
            pathways = np.atleast_2d(np.array(pathways, dtype=object)).tolist()

        added_pathways = []

        def identify_pway_type_and_parse_tuple_prn(pway, tuple_or_dict_str):
            """
            Determine whether pway is PROCESSING_PATHWAY or LEARNING_PATHWAY and, if it is the latter,
            parse tuple into pathway specification and LearningFunction.
            Return pathway type, pathway, and learning_function or None
            """
            learning_function = None

            if isinstance(pway, Pathway):
                pway = pway.pathway

            if (_is_node_spec(pway) or isinstance(pway, (list, set)) or
                    # Forgive use of tuple to specify a pathway, and treat as if it was a list spec
                    (isinstance(pway, tuple) and all(_is_pathway_entry_spec(n, ANY) for n in pathway))):
                pway_type = PROCESSING_PATHWAY
                if isinstance(pway, set):
                    pway = [pway]
                return pway_type, pway, None
            elif isinstance(pway, tuple):
                pway_type = LEARNING_PATHWAY
                if len(pway)!=2:
                    raise CompositionError(f"A tuple specified in the {pathways_arg_str}"
                                           f" has more than two items: {pway}")
                pway, learning_function = pway
                if not (_is_node_spec(pway) or isinstance(pway, (list, Pathway))):
                    raise CompositionError(f"The 1st item in {tuple_or_dict_str} specified in the "
                                           f" {pathways_arg_str} must be a node or a list: {pway}")
                if not (isinstance(learning_function, type) and issubclass(learning_function, LearningFunction)):
                    raise CompositionError(f"The 2nd item in {tuple_or_dict_str} specified in the "
                                           f"{pathways_arg_str} must be a LearningFunction: {learning_function}")
                return pway_type, pway, learning_function
            else:
                assert False, f"PROGRAM ERROR: arg to identify_pway_type_and_parse_tuple_prn in {self.name}" \
                              f"is not a Node, list or tuple: {pway}"

        # Validate items in pathways list and add to Composition using relevant add_linear_<> method.
        bad_entry_error_msg = f"Every item in the {pathways_arg_str} must be a " \
                              f"Node, list, set, tuple or dict; the following are not: "
        for pathway in pathways:
            pathway = pathway[0] if isinstance(pathway, list) and len(pathway) == 1 else pathway
            pway_name = None
            if isinstance(pathway, Pathway):
                pway_name = pathway.name
                pathway = pathway.pathway
            if _is_node_spec(pathway) or isinstance(pathway, (list, set, tuple)):
                if isinstance(pathway, set):
                    bad_entries = [repr(entry) for entry in pathway if not _is_node_spec(entry)]
                    if bad_entries:
                        raise CompositionError(f"{bad_entry_error_msg}{','.join(bad_entries)}")
                pway_type, pway, pway_learning_fct = identify_pway_type_and_parse_tuple_prn(pathway, f"a tuple")
            elif isinstance(pathway, dict):
                if len(pathway)!=1:
                    raise CompositionError(f"A dict specified in the {pathways_arg_str} "
                                           f"contains more than one entry: {pathway}.")
                pway_name, pway = list(pathway.items())[0]
                if not isinstance(pway_name, str):
                    raise CompositionError(f"The key in a dict specified in the {pathways_arg_str} must be a str "
                                           f"(to be used as its name): {pway_name}.")
                if _is_node_spec(pway) or isinstance(pway, (list, tuple, Pathway)):
                    pway_type, pway, pway_learning_fct = identify_pway_type_and_parse_tuple_prn(pway,
                                                                                                f"the value of a dict")
                else:
                    raise CompositionError(f"The value in a dict specified in the {pathways_arg_str} must be "
                                           f"a pathway specification (Node, list or tuple): {pway}.")
            else:
                raise CompositionError(f"{bad_entry_error_msg}{repr(pathway)}")

            context.source = ContextFlags.METHOD
            if pway_type == PROCESSING_PATHWAY:
                new_pathway = self.add_linear_processing_pathway(pathway=pway,
                                                                 name=pway_name,
                                                                 context=context)
            elif pway_type == LEARNING_PATHWAY:
                new_pathway = self.add_linear_learning_pathway(pathway=pway,
                                                               learning_function=pway_learning_fct,
                                                               name=pway_name,
                                                               context=context)
            else:
                assert False, f"PROGRAM ERROR: failure to determine pathway_type in add_pathways for {self.name}."

            added_pathways.append(new_pathway)

        return added_pathways

    @handle_external_context()
    def add_linear_processing_pathway(self, pathway, name:str=None, context=None, *args):
        """Add sequence of `Nodes <Composition_Nodes>` with optionally intercolated `Projections <Projection>`.

        .. _Composition_Add_Linear_Processing_Pathway:

        A Pathway is specified as a list, each element of which is either a `Node <Composition_Nodes>` or
        set of Nodes, possibly intercolated with specifications of `Projections <Projection>` between them.
        The Node(s) specified in each entry of the list project to the Node(s) specified in the next entry
        (see `Pathway_Specification` for details).

        .. note::
           Any specifications of the **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>`
           of a constructor for a `ControlMechanism` or the **monitor** argument in the constructor for an
           `ObjectiveMechanism` in the **objective_mechanism** `argument <ControlMechanism_ObjectiveMechanism>` of a
           ControlMechanism supercede any MappingProjections that would otherwise be created for them when specified
           in the **pathway** argument of add_linear_processing_pathway.

        Arguments
        ---------

        pathway : `Node <Composition_Nodes>`, list or `Pathway`
            specifies the `Nodes <Composition_Nodes>`, and optionally `Projections <Projection>`, used to construct a
            processing `Pathway <Pathway>`. Any standard form of `Pathway specification <Pathway_Specification>` can
            be used, however if a 2-item (Pathway, LearningFunction) tuple is used, the `LearningFunction` is ignored
            (this should be used with `add_linear_learning_pathway` if a `learning Pathway
            <Composition_Learning_Pathway>` is desired).  A `Pathway` object can also be used;  again, however, any
            learning-related specifications are ignored, as are its `name <Pathway.name>` if the **name** argument
            of add_linear_processing_pathway is specified.

        name : str
            species the name used for `Pathway`; supercedes `name <Pathway.name>` of `Pathway` object if it is has one.

        Returns
        -------

        `Pathway` :
            `Pathway` added to Composition.
        """

        from psyneulink.core.compositions.pathway import Pathway, _is_node_spec, _is_pathway_entry_spec

        def _get_spec_if_tuple(spec):
            return spec[0] if isinstance(spec, tuple) else spec

        nodes = []
        node_entries = []

        # If called internally, use its pathway_arg_str in error messages (in context.string)
        if context.source is not ContextFlags.COMMAND_LINE:
            pathway_arg_str = context.string
        # Otherwise, refer to call from this method
        else:
            pathway_arg_str = f"'pathway' arg for add_linear_procesing_pathway method of '{self.name}'"

        context.source = ContextFlags.METHOD
        context.string = pathway_arg_str

        pathway, pathway_name = self._parse_pathway(pathway, name, pathway_arg_str)

        # Verify that the pathway begins with a Node or set of Nodes
        if _is_node_spec(pathway[0]):
            # Use add_nodes so that node spec can also be a tuple with required_roles
            self.add_nodes(nodes=[pathway[0]], context=context)
            nodes.append(pathway[0])
            node_entries.append(pathway[0])
        # Or a set of Nodes
        elif isinstance(pathway[0], set):
            self.add_nodes(nodes=pathway[0], context=context)
            nodes.extend(pathway[0])
            node_entries.append(pathway[0])
        else:
            # 'MappingProjection has no attribute _name' error is thrown when pathway[0] is passed to the error msg
            raise CompositionError(f"First item in {pathway_arg_str} must be "
                                   f"a Node (Mechanism or Composition): {pathway}.")

        # Add all of the remaining nodes in the pathway
        for c in range(1, len(pathway)):
            # if the entry is for a Node (Mechanism, Composition or (Mechanism, NodeRole(s)) tuple), add it
            if _is_node_spec(pathway[c]):
                self.add_nodes(nodes=[pathway[c]],
                               context=context)
                nodes.append(pathway[c])
                node_entries.append(pathway[c])
            # If the entry is for a set of Nodes, add them
            elif isinstance(pathway[c], set) and all(_is_node_spec(entry) for entry in pathway[c]):
                self.add_nodes(nodes=pathway[c], context=context)
                nodes.extend(pathway[c])
                node_entries.append(pathway[c])

        # Then, delete any ControlMechanism that has its monitor_for_control attribute assigned
        #    and any ObjectiveMechanism that projects to a ControlMechanism,
        #    as well as any projections to them specified in the pathway;
        #    this is to avoid instantiating projections to them that might conflict with those
        #    instantiated by their constructors or, for a controller, _add_controller()
        items_to_delete = []
        for i, item in enumerate(pathway):
            if ((isinstance(item, ControlMechanism) and item.monitor_for_control)
                    or (isinstance(item, ObjectiveMechanism) and
                        set(self.get_roles_by_node(item)).intersection({NodeRole.CONTROL_OBJECTIVE,
                                                                          NodeRole.CONTROLLER_OBJECTIVE}))):
                items_to_delete.append(item)
                # Delete any projections to the ControlMechanism or ObjectiveMechanism specified in pathway
                if i>0 and _is_pathway_entry_spec(pathway[i - 1],PROJECTION):
                    items_to_delete.append(pathway[i - 1])
        for item in items_to_delete:
            if isinstance(item, ControlMechanism):
                arg_name = f'in the {repr(MONITOR_FOR_CONTROL)} of its constructor'
            else:
                arg_name = f'either in the {repr(MONITOR)} arg of its constructor, ' \
                           f'or in the {repr(MONITOR_FOR_CONTROL)} arg of its associated {ControlMechanism.__name__}'
            warnings.warn(f'No new {Projection.__name__}s were added to {item.name} that was included in '
                          f'the {pathway_arg_str}, since there were ones already specified {arg_name}.')
            del pathway[pathway.index(item)]

        # Then, loop through pathway and validate that the Mechanism-Projection relationships make sense
        # and add MappingProjection(s) where needed
        projections = []
        for c in range(1, len(pathway)):

            # NODE ENTRY ----------------------------------------------------------------------------------------
            def _get_node_specs_for_entry(entry, include_roles=None, exclude_roles=None):
                """Extract Nodes from any tuple specs and replace Compositions with their INPUT Nodes
                """
                nodes = []
                for node in entry:
                    # Extract Nodes from any tuple specs
                    node = _get_spec_if_tuple(node)
                    # Replace any nested Compositions with their INPUT Nodes
                    node = (self._get_nested_nodes_with_same_roles_at_all_levels(node, include_roles, exclude_roles)
                                if isinstance(node, Composition) else [node])
                    nodes.extend(node)
                return nodes

            # The current entry is a Node or a set of them:
            #  - if it is a set, list or array, leave as is, else place in set for consistency of processing below
            current_entry = pathway[c] if isinstance(pathway[c], (set, list, np.ndarray)) else {pathway[c]}
            if all(_is_node_spec(entry) for entry in current_entry):
                receivers = _get_node_specs_for_entry(current_entry, NodeRole.INPUT, NodeRole.TARGET)
                # The preceding entry is a Node or set of them:
                #  - if it is a set, list or array, leave as is, else place in set for consistnecy of processin below
                preceding_entry = (pathway[c - 1] if isinstance(pathway[c - 1], (set, list, np.ndarray))
                                   else {pathway[c - 1]})
                if all(_is_node_spec(sender) for sender in preceding_entry):
                    senders = _get_node_specs_for_entry(preceding_entry, NodeRole.OUTPUT)
                    projs = {self.add_projection(sender=s, receiver=r, allow_duplicates=False)
                            for r in receivers for s in senders}
                    if all(projs):
                        projs = projs.pop() if len(projs) == 1 else projs
                        projections.append(projs)

            # PROJECTION ENTRY --------------------------------------------------------------------------
            # Validate that it is between two nodes, then add the Projection;
            #    note: if Projection is already instantiated and valid, it is used as is;  if it is a set or list:
            #          - those are implemented between the corresponding pairs of sender and receiver Nodes
            #          - the list or set has a default Projection or matrix specification,
            #            that is used between all pairs of Nodes for which a Projection has not been specified

            # The current entry is a Projection specification or a list or set of them
            elif _is_pathway_entry_spec(pathway[c], PROJECTION):

                # Validate that Projection specification is not last entry
                if c == len(pathway) - 1:
                    raise CompositionError(f"The last item in the {pathway_arg_str} cannot be a Projection: "
                                           f"{pathway[c]}.")

                # Validate that entry is between two Nodes (or sets of Nodes)
                #     and get all pairings of sender and receiver nodes
                prev_entry = pathway[c - 1]
                next_entry = pathway[c + 1]
                if ((_is_node_spec(prev_entry) or isinstance(prev_entry, set))
                        and (_is_node_spec(next_entry) or isinstance(next_entry, set))):
                    senders = [_get_spec_if_tuple(sender) for sender in convert_to_list(prev_entry)]
                    receivers = [_get_spec_if_tuple(receiver) for receiver in convert_to_list(next_entry)]
                    node_pairs = list(itertools.product(senders,receivers))
                else:
                    raise CompositionError(f"A Projection specified in {pathway_arg_str} "
                                           f"is not between two Nodes: {pathway[c]}")

                # Convert specs in entry to list (embedding in one if matrix) for consistency of handling below
                all_proj_specs = [pathway[c]] if is_numeric(pathway[c]) else convert_to_list(pathway[c])

                # Get default Projection specification
                #  Must be a matrix spec, or a Projection with no sender or receiver specified
                #  If it is:
                #    - a single Projection, not in a set or list
                #    - appears only once in the pathways arg
                #    - it is preceded by only one sender Node and followed by only one receiver Node
                #  then treat as an individual Projection specification and not a default projection specification
                possible_default_proj_spec = [proj_spec for proj_spec in all_proj_specs
                                              if (is_matrix(proj_spec)
                                                  or (isinstance(proj_spec, Projection)
                                                      and proj_spec._initialization_status & ContextFlags.DEFERRED_INIT
                                                      and proj_spec._init_args[SENDER] is None
                                                      and proj_spec._init_args[RECEIVER] is None))]
                # Validate that there is no more than one default Projection specification
                if len(possible_default_proj_spec) > 1:
                    raise CompositionError(f"There is more than one matrix specification in the set of Projection "
                                           f"specifications for entry {c} of the {pathway_arg_str}: "
                                           f"{possible_default_proj_spec}.")
                # Get spec from list:
                spec = possible_default_proj_spec[0] if possible_default_proj_spec else None
                # If it appears only once on its own in the pathways arg and there is only one sender and one receiver
                #     consider it an individual Projection specification rather than a specification of the default
                if sum(isinstance(s, Projection) and s is spec for s in pathway) == len(senders) == len(receivers) == 1:
                    default_proj_spec = None
                    proj_specs = all_proj_specs
                else:
                    # Unpack if tuple spec, and assign feedback (with False as default)
                    default_proj_spec, feedback = (spec if isinstance(spec, tuple) else (spec, False))
                    # Get all specs other than default_proj_spec
                    # proj_specs = [proj_spec for proj_spec in all_proj_specs if proj_spec not in possible_default_proj_spec]
                    proj_specs = [proj_spec for proj_spec in all_proj_specs if proj_spec is not spec]

                # Collect all Projection specifications (to add to Composition at end)
                proj_set = []

                def handle_misc_errors(proj, error):
                    raise CompositionError(f"Bad Projection specification in {pathway_arg_str} ({proj}): "
                                           f"{str(error.error_value)}")

                def handle_duplicates(sender, receiver):
                    duplicate = [p for p in receiver.afferents if p in sender.efferents]
                    assert len(duplicate)==1, \
                        f"PROGRAM ERROR: Could not identify duplicate on DuplicateProjectionError " \
                        f"for {Projection.__name__} between {sender.name} and {receiver.name} " \
                        f"in call to {repr('add_linear_processing_pathway')} for {self.name}."
                    duplicate = duplicate[0]
                    warning_msg = f"Projection specified between {sender.name} and {receiver.name} " \
                                  f"in {pathway_arg_str} is a duplicate of one"
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
                    proj_set.append(self.add_projection(duplicate))

                # PARSE PROJECTION SPECIFICATIONS AND INSTANTIATE PROJECTIONS
                # IMPLEMENTATION NOTE:
                #    self.add_projection is called for each Projection
                #    to catch any duplicates with exceptions below

                # FIX: 4/9/22 - REFACTOR TO DO ANY SPECIFIED ASSIGNMENTS FIRST, AND THEN DEFAULT ASSIGNMENTS (IF ANY)
                if default_proj_spec is not None and not proj_specs:
                    # If there is a default specification and no other Projection specs,
                    #    use default to construct Projections for all node_pairs
                    for sender, receiver in node_pairs:
                        try:
                            # Default is a Projection
                            if isinstance(default_proj_spec, Projection):
                                # Copy so that assignments made to instantiated Projection don't affect default
                                projection = self.add_projection(projection=deepcopy(default_proj_spec),
                                                                 sender=sender,
                                                                 receiver=receiver,
                                                                 allow_duplicates=False,
                                                                 feedback=feedback)
                            else:
                                # Default is a matrix_spec
                                assert is_matrix(default_proj_spec), \
                                    f"PROGRAM ERROR: Expected {default_proj_spec} to be " \
                                    f"a matrix specification in {pathway_arg_str}."
                                projection = self.add_projection(projection=MappingProjection(sender=sender,
                                                                                              matrix=default_proj_spec,
                                                                                              receiver=receiver),
                                                                 allow_duplicates=False,
                                                                 feedback=feedback)
                            proj_set.append(projection)

                        except (InputPortError, ProjectionError, MappingError) as error:
                            handle_misc_errors(proj, error)
                        except DuplicateProjectionError:
                            handle_duplicates(sender, receiver)

                else:
                    # FIX: 4/9/22 - PUT THIS FIRST (BEFORE BLOCK JUST ABOVE) AND THEN ASSIGN TO ANY LEFT IN node_pairs
                    # Projections have been specified
                    for proj_spec in proj_specs:
                        try:
                            proj = _get_spec_if_tuple(proj_spec)
                            feedback = proj_spec[1] if isinstance(proj_spec, tuple) else False

                            if isinstance(proj, Projection):
                                # FIX 4/9/22 - TEST FOR DEFERRED INIT HERE (THAT IS NOT A default_proj_spec)
                                #              IF JUST SENDER OR RECEIVER, TREAT AS PER PORTS BELOW
                                # Validate that Projection is between a Node in senders and one in receivers
                                if proj._initialization_status & ContextFlags.DEFERRED_INIT:
                                    sender_node = senders[0]
                                    receiver_node = receivers[0]
                                else:
                                    sender_node = proj.sender.owner
                                    receiver_node = proj.receiver.owner
                                proj_set.append(self.add_projection(proj,
                                                                    sender = sender_node,
                                                                    receiver = receiver_node,
                                                                    allow_duplicates=False, feedback=feedback))
                                if default_proj_spec:
                                    # If there IS a default Projection specification, remove from node_pairs
                                    #   only the entry for the sender-receiver pair, so that the sender is assigned
                                    #   a default Projection to all other receivers (to which a Projection is not
                                    #   explicitly specified) and the receiver is assigned a default Projection from
                                    #   all other senders (from which a Projection is not explicitly specified).
                                    node_pairs = [pair for pair in node_pairs
                                                  if not all(node in pair for node in {sender_node, receiver_node})]
                                else:
                                    # If there is NOT a default Projection specification, remove from node_pairs
                                    #   all other entries with either the same sender OR receiver, so that neither
                                    #   the sender nor receiver are assigned any other default Projections.
                                    node_pairs = [pair for pair in node_pairs
                                                  if not any(node in pair for node in {sender_node, receiver_node})]

                            # FIX: 4/9/22 - SHOULD INCLUDE MECH SPEC (AND USE PRIMARY PORT) HERE:
                            elif isinstance(proj, Port):
                                # Implement default Projection (using matrix if specified) for all remaining specs
                                try:
                                    # FIX: 4/9/22 - INCLUDE TEST FOR DEFERRED_INIT WITH ONLY RECEIVER SPECIFIED
                                    if isinstance(proj, InputPort):
                                        for sender in senders:
                                            proj_set.append(self.add_projection(
                                                projection=MappingProjection(sender=sender, receiver=proj),
                                                allow_duplicates=False, feedback=feedback))
                                    # FIX: 4/9/22 - INCLUDE TEST FOR DEFERRED_INIT WITH ONLY SENDER SPECIFIED
                                    elif isinstance(proj, OutputPort):
                                        for receiver in receivers:
                                            proj_set.append(self.add_projection(
                                                projection=MappingProjection(sender=proj, receiver=receiver),
                                                allow_duplicates=False, feedback=feedback))
                                    # Remove from node_pairs all pairs involving the owner of the Port
                                    #   (since all Projections to or from it have been implemented)
                                    node_pairs = [pair for pair in node_pairs if (proj.owner not in pair)]
                                except (InputPortError, ProjectionError) as error:
                                    raise ProjectionError(str(error.error_value))

                        except (InputPortError, ProjectionError, MappingError) as error:
                            handle_misc_errors(proj, error)
                        except DuplicateProjectionError:
                            handle_duplicates(sender, receiver)

                    # FIX: 4/9/22 - REPLACE BELOW WITH CALL TO _assign_default_proj_spec(sender, receiver)
                    # If a default Projection is specified and any sender-receiver pairs remain, assign default
                    if default_proj_spec and node_pairs:
                        for sender, receiver in node_pairs:
                            try:
                                p = self.add_projection(projection=deepcopy(default_proj_spec),
                                                        sender=sender,
                                                        receiver=receiver,
                                                        allow_duplicates=False,
                                                        feedback=feedback)
                                proj_set.append(p)
                            except (InputPortError, ProjectionError, MappingError) as error:
                                handle_misc_errors(proj, error)
                            except DuplicateProjectionError:
                                handle_duplicates(sender, receiver)

                # If there is a single Projection, extract it from list and append as Projection
                # IMPLEMENTATION NOTE:
                #    this is to support calls to add_learing_processing_pathway by add_learning_<> methods
                #    that do not yet support a list or set of Projection specifications
                if len(proj_set) == 1:
                    projections.append(proj_set[0])
                else:
                    projections.append(proj_set)

            # BAD PATHWAY ENTRY: contains neither Node nor Projection specification(s)
            else:
                assert False, f"PROGRAM ERROR : An entry in {pathway_arg_str} is not a Node (Mechanism " \
                              f"or Composition) or a Projection nor a set of either: {repr(pathway[c])}."

        # Finally, clean up any tuple specs
        for i, n_e in enumerate(node_entries):
            for n in convert_to_list(n_e):
                if isinstance(n, tuple):
                    nodes[i] = nodes[i][0]
        # interleave (sets of) Nodes and (sets or lists of) Projections
        explicit_pathway = [node_entries[0]]
        for i in range(len(projections)):
            explicit_pathway.append(projections[i])
            explicit_pathway.append(node_entries[i + 1])

        # If pathway is an existing one, return that
        existing_pathway = next((p for p in self.pathways if explicit_pathway==p.pathway), None)
        if existing_pathway:
            warnings.warn(f"Pathway specified in {pathway_arg_str} already exists in {self.name}: {pathway}; "
                          f"it will be ignored.")
            return existing_pathway
        # If the explicit pathway is shorter than the one specified, then need to do more checking
        elif len(explicit_pathway) < len(pathway):
            # Pathway without all Projections specified has same nodes in same order as existing one
            existing_pathway = next((p for p in self.pathways
                                     if [item for p in self.pathways for item in p.pathway
                                         if not isinstance(item, Projection)]), None)
            # Shorter because Projections generated for unspecified ones duplicated existing ones & were suppressed
            if existing_pathway:
                warnings.warn(f"Pathway specified in {pathway_arg_str} has same Nodes in same order as "
                              f"one already in {self.name}: {pathway}; it will be ignored.")
                return existing_pathway
            #
            # Shorter because it contained one or more ControlMechanisms with monitor_for_control specified.
            elif explicit_pathway == [m for m in pathway
                                      if not (isinstance(m, ControlMechanism)
                                              or (isinstance(m, tuple) and isinstance(m[0], ControlMechanism)))]:
                pass
            else:
                # Otherwise, something has gone wrong
                assert False, \
                    f"PROGRAM ERROR: Bad pathway specification for {self.name} in {pathway_arg_str}: {pathway}."

        pathway = Pathway(pathway=explicit_pathway,
                          composition=self,
                          name=pathway_name,
                          context=context)
        self.pathways.append(pathway)

        self._analyze_graph(context)

        return pathway

    # endregion PROCESSING PATHWAYS

    # region ------------------------------------ LEARNING -------------------------------------------------------------

    @handle_external_context()
    def add_linear_learning_pathway(self,
                                    pathway,
                                    learning_function:LearningFunction,
                                    loss_function=None,
                                    learning_rate:tc.any(int,float)=0.05,
                                    error_function=LinearCombination,
                                    learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=AFTER,
                                    name:str=None,
                                    context=None):
        """Implement learning pathway (including necessary `learning components <Composition_Learning_Components>`.

        Generic method for implementing a learning pathway.  Calls `add_linear_processing_pathway` to implement
        the processing components including, if necessary, the MappingProjections between Mechanisms.  All of the
        MappingProjections (whether specified or created) are subject to learning (and are assigned as the
        `learned_projection <LearningMechanism.learned_projection>` attribute of the `LearningMechanisms
        <LeaningMechanisms>` created for the pathway.

        If **learning_function** is a sublcass of `LearningFunction <LearningFunctions>`, a class-specific
        `learning method <Composition_Learning_Methods>` is called.  Some may allow the error_function
        to be specified, in which case it must be compatible with the class of LearningFunction specified.

        If **learning_function** is a custom function, it is assigned to all of the `LearningMechanisms
        <LearningMechanism>` created for the MappingProjections in the pathway.  A `ComparatorMechanism` is
        created to compute the error for the pathway, and assigned the function specified in **error_function**,
        which must be compatible with **learning_function**.

        See `Composition_Learning` for a more detailed description of how learning is implemented in a
        Composition, including the `learning components <Composition_Learning_Components>` that are created,
        as well as other `learning methods <Composition_Learning_Methods>` that can be used to implement specific
        algorithms.

        The `learning components <Composition_Learning_Components>` created are placed in a dict the following entries:
            *OUTPUT_MECHANISM*: `ProcessingMechanism` (assigned to `output <Pathway.output>`
            *TARGET_MECHANISM*: `ProcessingMechanism` (assigned to `target <Pathway.target>`
            *OBJECTIVE_MECHANISM*: `ComparatorMechanism` (assigned to `learning_objective <Pathway.learning_objective>`
            *LEARNING_MECHANISMS*: `LearningMechanism` or list[`LearningMechanism`]
            *LEARNING_FUNCTION*: `LearningFunction` used by all LEARNING_MECHSNISMS in the `Pathway`
            *LEARNED_PROJECTIONS*: `Projection <Projection>` or list[`Projections <Projection>`]
        that is assigned to the `learning_components <Pathway.learning_components>` attribute of the `Pathway`
        returned.

        Arguments
        ---------

        pathway : List or `Pathway`
            specifies the `learning Pathway <Composition_Learning_Pathway>` for which the `Projections <Projections>`
            will be learned;  can be specified as a `Pathway` or as a list of `Nodes <Composition_Nodes>` and,
            optionally, Projections between them (see `list <Pathway_Specification_List>`).

        learning_function : LearningFunction
            specifies the type of `LearningFunction` to use for the `LearningMechanism` constructued for each
            `MappingProjection` in the **pathway**.

        loss_function : MSE or SSE : default None
            specifies the loss function used if `BackPropagation` is specified as the **learning_function**
            (see `add_backpropagation_learning_pathway <Composition.add_backpropagation_learning_pathway>`).

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
            in each `TRIAL <TimeScale.TRIAL>` when the Composition executes;  it is assigned as the default value for
            the `learning_enabled <LearningMechanism.learning_enabled>` attribute of the `LearningMechanism
            <LearningMechanism>` in the pathway, and its `LearningProjection` (see `learning_enabled
            <LearningMechanism.learning_enabled>` for meaning of values).

        name : str :
            species the name used for `Pathway`; supercedes `name <Pathway.name>` of `Pathway` object if it is has one.

        Returns
        --------

        `Pathway` :
            `learning Pathway` <Composition_Learning_Pathway>` added to the Composition.

        """

        from psyneulink.core.compositions.pathway import Pathway, PathwayRole

        # If called from add_pathways(), use its pathway_arg_str
        if context.source == ContextFlags.METHOD:
            pathway_arg_str = context.string
        # Otherwise, refer to call from this method
        else:
            pathway_arg_str = f"'pathway' arg for add_linear_procesing_pathway method of {self.name}"
        context.source = ContextFlags.METHOD
        context.string = pathway_arg_str

        # Deal with Pathway() specifications
        if isinstance(pathway, Pathway):
            pathway_name = name or pathway.name
            pathway = pathway.pathway
        else:
            pathway_name = name

        # Make sure pathways is not a (<pathway spec>, LearningFunction) tuple that conflicts with learning_function
        if isinstance(pathway,tuple) and pathway[1] is not learning_function:
            raise CompositionError(f"Specification in {pathway_arg_str} contains a tuple that specifies a different "
                                   f"{LearningFunction.__name__} ({pathway[1].__name__}) than the one specified in "
                                   f"its 'learning_function' arg ({learning_function.__name__}).")

        # Preserve existing NodeRole.OUTPUT status for any non-learning-related nodes
        for node in self.get_nodes_by_role(NodeRole.OUTPUT):
            if not any(node for node in [pathway for pathway in self.pathways
                                     if PathwayRole.LEARNING in pathway.roles]):
                self._add_required_node_role(node, NodeRole.OUTPUT, context)

        # Handle BackPropgation specially, since it is potentially multi-layered
        if isinstance(learning_function, type) and issubclass(learning_function, BackPropagation):
            return self._create_backpropagation_learning_pathway(pathway,
                                                                 learning_rate,
                                                                 error_function,
                                                                 loss_function,
                                                                 learning_update,
                                                                 name=pathway_name,
                                                                 context=context)

        # If BackPropagation is not specified, then the learning pathway is "one-layered"
        #   (Mechanism -> learned_projection -> Mechanism) with only one LearningMechanism, Target and Comparator

        # Processing Components
        try:
            input_source, output_source, learned_projection = \
                self._unpack_processing_components_of_learning_pathway(pathway)
        except CompositionError as e:
            raise CompositionError(e.error_value.replace('this method',
                                                         f'{learning_function.__name__} {LearningFunction.__name__}'))

        # Add required role before calling add_linear_process_pathway so NodeRole.OUTPUTS are properly assigned
        self._add_required_node_role(output_source, NodeRole.OUTPUT, context)

        learning_pathway = self.add_linear_processing_pathway(pathway=[input_source, learned_projection, output_source],
                                                              name=pathway_name,
                                                              # context=context)
                                                              context=context)

        # Learning Components
        target, comparator, learning_mechanism = self._create_learning_related_mechanisms(input_source,
                                                                                          output_source,
                                                                                          error_function,
                                                                                          learning_function,
                                                                                          learned_projection,
                                                                                          learning_rate,
                                                                                          learning_update)

        # Suppress warning regarding no efferent projections from Comparator (since it is a TERMINAL node)
        for s in comparator.output_ports:
            s.parameters.require_projection_in_composition.set(False,
                                                               override=True)
        # Add nodes to Composition
        self.add_nodes([(target, NodeRole.TARGET),
                        (comparator, NodeRole.LEARNING_OBJECTIVE),
                         learning_mechanism],
                       required_roles=NodeRole.LEARNING,
                       context=context)

        # Create Projections to and among learning-related Mechanisms and add to Composition
        learning_related_projections = self._create_learning_related_projections(input_source,
                                                                                 output_source,
                                                                                 target,
                                                                                 comparator,
                                                                                 learning_mechanism)
        self.add_projections(learning_related_projections)

        # Create Projection to learned Projection and add to Composition
        learning_projection = self._create_learning_projection(learning_mechanism, learned_projection)
        self.add_projection(learning_projection, learning_projection=True)

        # FIX 5/8/20: WHY IS LEARNING_MECHANSIMS ASSIGNED A SINGLE MECHANISM?
        # Wrap up and return
        learning_related_components = {OUTPUT_MECHANISM: output_source,
                                       TARGET_MECHANISM: target,
                                       OBJECTIVE_MECHANISM: comparator,
                                       LEARNING_MECHANISMS: learning_mechanism,
                                       LEARNED_PROJECTIONS: learned_projection,
                                       LEARNING_FUNCTION: learning_function}
        learning_pathway.learning_components = learning_related_components
        # Update graph in case method is called again
        self._analyze_graph()
        return learning_pathway


    def add_reinforcement_learning_pathway(self,
                                           pathway,
                                           learning_rate=0.05,
                                           error_function=None,
                                           learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=ONLINE,
                                           name:str=None):
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
            in each `TRIAL <TimeScale.TRIAL>` when the Composition executes;  it is assigned as the default value for
            the `learning_enabled <LearningMechanism.learning_enabled>` attribute of the `LearningMechanism
            <LearningMechanism>` in the pathway, and its `LearningProjection` (see `learning_enabled
            <LearningMechanism.learning_enabled>` for meaning of values).

        name : str :
            species the name used for `Pathway`; supercedes `name <Pathway.name>` of `Pathway` object if it is has one.

        Returns
        --------

        `Pathway` :
            Reinforcement `learning Pathway` <Composition_Learning_Pathway>` added to the Composition.

        """
        return self.add_linear_learning_pathway(pathway,
                                                learning_rate=learning_rate,
                                                learning_function=Reinforcement,
                                                error_function=error_function,
                                                learning_update=learning_update,
                                                name=name)

    def add_td_learning_pathway(self,
                                pathway,
                                learning_rate=0.05,
                                error_function=None,
                                learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=ONLINE,
                                name:str=None):
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
            in each `TRIAL <TimeScale.TRIAL>` when the Composition executes;  it is assigned as the default value for
            the `learning_enabled <LearningMechanism.learning_enabled>` attribute of the `LearningMechanism
            <LearningMechanism>` in the pathway, and its `LearningProjection` (see `learning_enabled
            <LearningMechanism.learning_enabled>` for meaning of values).

        name : str :
            species the name used for `Pathway`; supercedes `name <Pathway.name>` of `Pathway` object if it is has one.

        Returns
        --------

        `Pathway` :
            TD Reinforcement `learning Pathway` <Composition_Learning_Pathway>` added to the Composition.

        """
        return self.add_linear_learning_pathway(pathway,
                                                learning_rate=learning_rate,
                                                learning_function=TDLearning,
                                                learning_update=learning_update,
                                                name=name)

    def add_backpropagation_learning_pathway(self,
                                             pathway,
                                             learning_rate=0.05,
                                             error_function=None,
                                             loss_function:tc.enum(MSE,SSE)=MSE,
                                             learning_update:tc.optional(tc.any(bool, tc.enum(ONLINE, AFTER)))=AFTER,
                                             name:str=None):
        """Convenience method that calls `add_linear_learning_pathway` with **learning_function**=`Backpropagation`

        Arguments
        ---------
        pathway: List
            specifies nodes of the `Pathway` for the `learning pathway <Composition_Learning_Pathway>` (see
            `add_linear_processing_pathway` for details of specification).  Any `MappingProjections
            <MappingProjection>` specified or constructed for the Pathway are assigned as `learned_projections`.

        learning_rate : float : default 0.05
            specifies the `learning_rate <Backpropagation.learning_rate>` used for the `Backpropagation` function of
            the `LearningMechanisms <LearningMechanism>` in the **pathway**.

        error_function : function : default LinearCombination
            specifies the function assigned to `ComparatorMechanism` used to compute the error from the target and the
            output (`value <Mechanism_Base.value>`) of the `TARGET` (last) Mechanism in the **pathway**).

        loss_function : MSE or SSE : default MSE
            specifies the loss function used in computing the error term;
            MSE = mean squared error, and SSE = sum squared error.

        learning_update : Optional[bool|ONLINE|AFTER] : default AFTER
            specifies when the `matrix <MappingProjection.matrix>` parameters of the `learned_projections` are updated
            in each `TRIAL <TimeScale.TRIAL>` when the Composition executes;  it is assigned as the default value for
            the `learning_enabled <LearningMechanism.learning_enabled>` attribute of the `LearningMechanisms
            <LearningMechanism>` in the pathway, and their `LearningProjections <LearningProjection>`
            (see `learning_enabled <LearningMechanism.learning_enabled>` for meaning of values).

        name : str :
            species the name used for `Pathway`; supercedes `name <Pathway.name>` of `Pathway` object if it is has one.

        Returns
        --------

        `Pathway` :
            BackPropagation `learning Pathway` <Composition_Learning_Pathway>` added to the Composition.

        """
        return self.add_linear_learning_pathway(pathway,
                                                learning_rate=learning_rate,
                                                learning_function=BackPropagation,
                                                loss_function=loss_function,
                                                error_function=error_function,
                                                learning_update=learning_update,
                                                name=name)

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
        if len(processing_pathway) == 3 and isinstance(processing_pathway[1], MappingProjection):
            input_source, learned_projection, output_source = processing_pathway
        elif len(processing_pathway) == 2:
            input_source, output_source = processing_pathway
            learned_projection = MappingProjection(sender=input_source, receiver=output_source)
        else:
            raise CompositionError(f"Too many Nodes in learning pathway: {processing_pathway}. "
                                   f"Only single-layer learning is supported by this method. "
                                   f"Use {BackPropagation.__name__} LearningFunction or "
                                   f"see AutodiffComposition for other learning models.")
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

        # ONLY DO THIS IF ONE DOESN'T ALREADY EXIST (?pass in argument determining this?)
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
        """Creates *TARGET_MECHANISM*, `ComparatorMechanism` and `LearningMechanism` for RL and TD learning"""

        if isinstance(learning_function, type):
            if issubclass(learning_function, TDLearning):
                creation_method = self._create_td_related_mechanisms
            elif issubclass(learning_function, Reinforcement):
                creation_method = self._create_rl_related_mechanisms
            else:
                raise CompositionError(f"'learning_function' argument for add_linear_learning_pathway "
                                       f"({learning_function}) must be a class of {LearningFunction.__name__}")

            target_mechanism, objective_mechanism, learning_mechanism  = creation_method(input_source,
                                                                                          output_source,
                                                                                          error_function,
                                                                                          learned_projection,
                                                                                          learning_rate,
                                                                                          learning_update)

        elif is_function_type(learning_function):
            target_mechanism = ProcessingMechanism(name='Target')
            objective_mechanism = ComparatorMechanism(name='Comparator',
                                                      sample={NAME: SAMPLE,
                                                              VARIABLE: [0.], WEIGHT: -1},
                                                      target={NAME: TARGET,
                                                              VARIABLE: [0.]},
                                                      function=error_function,
                                                      output_ports=[OUTCOME, MSE],
                                                      )
            learning_mechanism = LearningMechanism(
                                    function=learning_function(
                                                         default_variable=[input_source.output_ports[0].value,
                                                                           output_source.output_ports[0].value,
                                                                           objective_mechanism.output_ports[0].value],
                                                         learning_rate=learning_rate),
                                    default_variable=[input_source.output_ports[0].value,
                                                      output_source.output_ports[0].value,
                                                      objective_mechanism.output_ports[0].value],
                                    error_sources=objective_mechanism,
                                    learning_enabled=learning_update,
                                    in_composition=True,
                                    name="Learning Mechanism for " + learned_projection.name)

            objective_mechanism.modulatory_mechanism = learning_mechanism

        else:
            raise CompositionError(f"'learning_function' argument of add_linear_learning_pathway "
                                   f"({learning_function}) must be a class of {LearningFunction.__name__} or a "
                                   f"learning-compatible function")

        learning_mechanism.output_ports[ERROR_SIGNAL].parameters.require_projection_in_composition.set(False,
                                                                                                         override=True)
        return target_mechanism, objective_mechanism, learning_mechanism

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
        """Construct LearningProjections from LearningMechanisms to learned_projections in a learning pathway"""

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

        objective_mechanism = ComparatorMechanism(name='Comparator',
                                                  sample={NAME: SAMPLE,
                                                          VARIABLE: [0.], WEIGHT: -1},
                                                  target={NAME: TARGET,
                                                          VARIABLE: [0.]},
                                                  function=error_function,
                                                  output_ports=[OUTCOME, MSE],
                                                  )

        learning_mechanism = \
            LearningMechanism(function=Reinforcement(default_variable=[input_source.output_ports[0].value,
                                                                       output_source.output_ports[0].value,
                                                                       objective_mechanism.output_ports[0].value],
                                                     learning_rate=learning_rate),
                              default_variable=[input_source.output_ports[0].value,
                                                output_source.output_ports[0].value,
                                                objective_mechanism.output_ports[0].value],
                              error_sources=objective_mechanism,
                              learning_enabled=learning_update,
                              in_composition=True,
                              name="Learning Mechanism for " + learned_projection.name)

        objective_mechanism.modulatory_mechanism = learning_mechanism

        return target_mechanism, objective_mechanism, learning_mechanism

    def _create_td_related_mechanisms(self,
                                      input_source,
                                      output_source,
                                      error_function,
                                      learned_projection,
                                      learning_rate,
                                      learning_update):

        target_mechanism = ProcessingMechanism(name='Target',
                                               default_variable=output_source.defaults.value)

        objective_mechanism = PredictionErrorMechanism(name='PredictionError',
                                                        sample={NAME: SAMPLE,
                                                                VARIABLE: np.zeros_like(output_source.output_ports[0].defaults.value)},
                                                        target={NAME: TARGET,
                                                                VARIABLE: np.zeros_like(output_source.output_ports[0].defaults.value)},
                                                        function=PredictionErrorDeltaFunction(gamma=1.0))

        learning_mechanism = LearningMechanism(function=TDLearning(learning_rate=learning_rate),
                                               default_variable=[input_source.output_ports[0].defaults.value,
                                                                 output_source.output_ports[0].defaults.value,
                                                                 objective_mechanism.output_ports[0].defaults.value],
                                               error_sources=objective_mechanism,
                                               learning_enabled=learning_update,
                                               in_composition=True,
                                               name="Learning Mechanism for " + learned_projection.name)

        return target_mechanism, objective_mechanism, learning_mechanism

    def _create_backpropagation_learning_pathway(self,
                                                 pathway,
                                                 learning_rate=0.05,
                                                 error_function=None,
                                                 loss_function=MSE,
                                                 learning_update=AFTER,
                                                 name=None,
                                                 context=None):

        # FIX: LEARNING CONSOLIDATION - Can get rid of this:
        if not error_function:
            error_function = LinearCombination()
        if not loss_function:
            loss_function = MSE

        # Add pathway to graph and get its full specification (includes all ProcessingMechanisms and MappingProjections)
        # Pass ContextFlags.INITIALIZING so that it can be passed on to _analyze_graph() and then
        #    _check_for_projection_assignments() in order to ignore checks for require_projection_in_composition
        context.string = f"'pathway' arg for add_backpropagation_learning_pathway method of {self.name}"
        learning_pathway = self.add_linear_processing_pathway(pathway, name, context)
        processing_pathway = learning_pathway.pathway

        path_length = len(processing_pathway)

        # Pathway length must be >=3 (Mechanism, Projection, Mechanism)
        if path_length >= 3:
            # get the "terminal_sequence" --
            # the last 2 nodes in the back prop pathway and the projection between them
            # these components are processed separately because
            # they inform the construction of the Target and Comparator mechs
            terminal_sequence = processing_pathway[path_length - 3: path_length]
        else:
            raise CompositionError(f"Backpropagation pathway specification "
                                   f"does not have enough components: {pathway}.")

        # Unpack and process terminal_sequence:
        input_source, learned_projection, output_source = terminal_sequence

        # If pathway includes existing terminal_sequence for the output_source, use that
        if output_source in self._terminal_backprop_sequences:

            # FIX CROSSED_PATHWAYS 7/28/19 [JDC]:
            #  THIS SHOULD BE INTEGRATED WITH CALL TO _create_terminal_backprop_learning_components
            #  ** NEED TO CHECK WHETHER LAST NODE IN THE SEQUENCE IS TERMINAL AND IF SO:
            #     ASSIGN USING: self._add_required_node_role(output_source, NodeRole.OUTPUT)
            # If learned_projection already has a LearningProjection (due to pathway overlap),
            #    use those terminal sequence components
            if (learned_projection.has_learning_projection
                    and any([lp for lp in learned_projection.parameter_ports[MATRIX].mod_afferents
                             if lp in self.projections])):
                target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
                comparator = self._terminal_backprop_sequences[output_source][OBJECTIVE_MECHANISM]
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
                                                                       learning_update,
                                                                       context)
            sequence_end = path_length - 3

        # # FIX: ALTERNATIVE IS TO TEST WHETHER IT PROJECTS TO ANY MECHANISMS WITH LEARNING ROLE
        # Otherwise, if output_source already projects to a LearningMechanism in the current Composition,
        #     integrate with existing sequence
        elif any((isinstance(p.receiver.owner, LearningMechanism)
                  and p.receiver.owner in self.learning_components)
                 for p in output_source.efferents):
            # Set learning_mechanism to the one to which output_source projects
            learning_mechanism = next((p.receiver.owner for p in output_source.efferents
                                       if isinstance(p.receiver.owner, LearningMechanism)))
            # # Use existing target and comparator to learning_mechanism for Mechanism to which output_source project
            # target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
            # comparator = self._terminal_backprop_sequences[output_source][OBJECTIVE_MECHANISM]
            target = None
            comparator = None
            sequence_end = path_length - 1

        # Otherwise create terminal_sequence for the sequence,
        #    and eliminate existing terminal_sequences previously created for Mechanisms now in the pathway
        else:
            # Eliminate existing comparators and targets for Mechanisms now in the pathway that were output_sources
            #   (i.e., ones that belong to previously-created sequences that overlap with the current one)
            for pathway_mech in [m for m in processing_pathway if isinstance(m, Mechanism)]:

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
                                                                   learning_update,
                                                                   context)
            self._terminal_backprop_sequences[output_source] = {LEARNING_MECHANISM: learning_mechanism,
                                                                TARGET_MECHANISM: target,
                                                                OBJECTIVE_MECHANISM: comparator}
            self._add_required_node_role(processing_pathway[-1], NodeRole.OUTPUT, context)

            sequence_end = path_length - 3

        # loop backwards through the rest of the processing_pathway to create and connect
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
                                                                                        learning_update,
                                                                                        context)
            learning_mechanisms.append(learning_mechanism)
            learned_projections.append(learned_projection)

        # Add error_signal projections to any learning_mechanisms that are now dependent on the new one
        for lm in learning_mechanisms:
            if lm.dependent_learning_mechanisms:
                projections = self._add_error_projection_to_dependent_learning_mechs(lm, context)
                self.add_projections(projections)

        # Suppress "no efferent connections" warning for:
        #    - error_signal OutputPort of last LearningMechanism in sequence
        #    - comparator
        learning_mechanisms[-1].output_ports[ERROR_SIGNAL].parameters.require_projection_in_composition.set(
            False,
            override=True
        )
        if comparator:
            for s in comparator.output_ports:
                s.parameters.require_projection_in_composition.set(False,
                                                                   override=True)

        learning_related_components = {OUTPUT_MECHANISM: pathway[-1],
                                       TARGET_MECHANISM: target,
                                       OBJECTIVE_MECHANISM: comparator,
                                       LEARNING_MECHANISMS: learning_mechanisms,
                                       LEARNED_PROJECTIONS: learned_projections,
                                       LEARNING_FUNCTION: BackPropagation}

        learning_pathway.learning_components = learning_related_components

        # Update graph in case method is called again
        self._analyze_graph()

        return learning_pathway

    def infer_backpropagation_learning_pathways(self):
        """Convenience method that automatically creates backpropapagation learning pathways for every
        Input Node --> Output Node pathway
        """
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
                    # we only consider input -> projection -> ... -> output pathways
                    # (since we can't learn on only one mechanism)
                    if len(p) >= 3:
                        pathways.append(p)
                    continue
                for projection, efferent_node in [(p, p.receiver.owner) for p in curr_node.efferents]:
                    if (not hasattr(projection,'learnable')) or (projection.learnable is False):
                        continue
                    prev[efferent_node] = projection
                    prev[projection] = curr_node
                    queue.append(efferent_node)
            return pathways

        pathways = [p for n in self.get_nodes_by_role(NodeRole.INPUT) if
                    NodeRole.TARGET not in self.get_roles_by_node(n) for p in bfs(n)]
        for pathway in pathways:
            self.add_backpropagation_learning_pathway(pathway=pathway)

    def _create_terminal_backprop_learning_components(self,
                                                      input_source,
                                                      output_source,
                                                      error_function,
                                                      loss_function,
                                                      learned_projection,
                                                      learning_rate,
                                                      learning_update,
                                                      context):
        """Create ComparatorMechanism, LearningMechanism and LearningProjection for Component in learning Pathway"""

        # target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
        # comparator = self._terminal_backprop_sequences[output_source][OBJECTIVE_MECHANISM]
        # learning_mechanism = self._terminal_backprop_sequences[output_source][LEARNING_MECHANISM]

        # If target and comparator already exist (due to overlapping pathway), use those
        try:
            target_mechanism = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
            objective_mechanism = self._terminal_backprop_sequences[output_source][OBJECTIVE_MECHANISM]

        # Otherwise, create new ones
        except KeyError:
            target_mechanism = ProcessingMechanism(name='Target',
                                                   default_variable=output_source.output_ports[0].value)
            objective_mechanism = ComparatorMechanism(name='Comparator',
                                                      target={NAME: TARGET,
                                                              VARIABLE: target_mechanism.output_ports[0].value},
                                                      sample={NAME: SAMPLE,
                                                              VARIABLE: output_source.output_ports[0].value,
                                                              WEIGHT: -1},
                                                      function=error_function,
                                                      output_ports=[OUTCOME, MSE],
                                                      )

        learning_function = BackPropagation(default_variable=[input_source.output_ports[0].value,
                                                              output_source.output_ports[0].value,
                                                              objective_mechanism.output_ports[0].value],
                                            activation_derivative_fct=output_source.function.derivative,
                                            learning_rate=learning_rate,
                                            loss_function=loss_function)

        learning_mechanism = LearningMechanism(function=learning_function,
                                               default_variable=[input_source.output_ports[0].value,
                                                                 output_source.output_ports[0].value,
                                                                 objective_mechanism.output_ports[0].value],
                                               error_sources=objective_mechanism,
                                               learning_enabled=learning_update,
                                               in_composition=True,
                                               name="Learning Mechanism for " + learned_projection.name)

        objective_mechanism.modulatory_mechanism = learning_mechanism

        self.add_nodes(nodes=[(target_mechanism, NodeRole.TARGET),
                              (objective_mechanism, NodeRole.LEARNING_OBJECTIVE),
                              learning_mechanism],
                       required_roles=NodeRole.LEARNING,
                       context=context)

        learning_related_projections = self._create_learning_related_projections(input_source,
                                                                                 output_source,
                                                                                 target_mechanism,
                                                                                 objective_mechanism,
                                                                                 learning_mechanism)
        self.add_projections(learning_related_projections)

        learning_projection = self._create_learning_projection(learning_mechanism, learned_projection)
        self.add_projection(learning_projection, feedback=True)


        return target_mechanism, objective_mechanism, learning_mechanism

    def _create_non_terminal_backprop_learning_components(self,
                                                          input_source,
                                                          output_source,
                                                          learned_projection,
                                                          learning_rate,
                                                          learning_update,
                                                          context):

        # Get existing LearningMechanism if one exists (i.e., if this is a crossing point with another pathway)
        learning_mechanism = \
            next((lp.sender.owner for lp in learned_projection.parameter_ports[MATRIX].mod_afferents
                  if isinstance(lp, LearningProjection)),
                 None)

        # If learning_mechanism exists:
        #    error_sources will be empty (as they have been dealt with in self._get_back_prop_error_sources
        #    error_projections will contain list of any created to be added to the Composition below
        if learning_mechanism:
            error_sources, error_projections = self._get_back_prop_error_sources(output_source,
                                                                                 learning_mechanism,
                                                                                 context)
        # If learning_mechanism does not yet exist:
        #    error_sources will contain ones needed to create learning_mechanism
        #    error_projections will be empty since they can't be created until the learning_mechanism is created below;
        #    they will be created (using error_sources) when, and determined after learning_mechanism is created below
        else:
            error_sources, error_projections = self._get_back_prop_error_sources(output_source, context=context)
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

        self.add_node(learning_mechanism, required_roles=NodeRole.LEARNING, context=context)
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

    def _get_back_prop_error_sources(self, receiver_activity_mech, learning_mech=None, context=None):
        # FIX CROSSED_PATHWAYS [JDC]:  GENERALIZE THIS TO HANDLE COMPARATOR/TARGET ASSIGNMENTS IN BACKPROP
        #                              AND THEN TO HANDLE ALL FORMS OF LEARNING (AS BELOW)
        #  REFACTOR TO DEAL WITH CROSSING PATHWAYS (?CREATE METHOD ON LearningMechanism TO DO THIS?):
        #  1) Determine whether this is a terminal sequence:
        #     - use arg passed in or determine from context
        #       (see current implementation in add_backpropagation_learning_pathway)
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
                                      context=context),
                            context=context)[0]
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

    def _get_backprop_error_projections(self, learning_mech, receiver_activity_mech, context):
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
                                                                  context=context),
                                                        context=context)
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

    def _add_error_projection_to_dependent_learning_mechs(self, error_source, context=None):
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
                                                              context=context),
                                                    context=context)
                projections.append(error_signal_input_port[0].path_afferents[0])
                # projections.append(MappingProjection(sender=error_source.output_ports[ERROR_SIGNAL],
                #                                      receiver=error_signal_input_port[0]))
        return projections

    def _get_deeply_nested_aux_projections(self, node):
        deeply_nested_projections = {}
        if hasattr(node, 'aux_components'):
            aux_projections = {}
            for i in node.aux_components:
                if hasattr(i, '__iter__') and isinstance(i[0], Projection):
                    aux_projections[i] = i[0]
                elif isinstance(i, Projection):
                    aux_projections[i] = i
            nested_nodes = self._get_nested_nodes()
            for spec, proj in aux_projections.items():
                # FIX: TREATMENT OF RECEIVERS SEEMS TO DEAL WITH ONLY RECEIVERS IN COMPS NESTED MORE THAN ONE LEVEL DEEP
                #      REMOVING "if not i[1] in self.nodes" crashes in test_multilevel_control
                if ((proj.sender.owner not in self.nodes
                     and proj.sender.owner in [i[0] for i in nested_nodes])
                        or (proj.receiver.owner not in self.nodes
                            and proj.receiver.owner in [i[0] for i in nested_nodes if not i[1] in self.nodes])):
                    deeply_nested_projections[spec] = proj
        return deeply_nested_projections

    # endregion LEARNING PATHWAYS

    # endregion PATHWAYS

    # ******************************************************************************************************************
    # region ------------------------------------- CONTROL -------------------------------------------------------------
    # ******************************************************************************************************************

    @handle_external_context()
    def add_controller(self, controller:ControlMechanism, context=None):
        """
        Add a `ControlMechanism` as the `controller <Composition.controller>` of the Composition.

        This gives the ControlMechanism access to the `Composition`'s `evaluate <Composition.evaluate>` method. This
        allows subclasses of ControlMechanism that can use this (such as `OptimizationControlMechanism`) to execute
        `simulations <OptimizationControlMechanism_Execution>` of the Composition (that is, executions in an
        `execution context <Composition_Execution_Context>` separate from the one used by the `execution method
        <Composition_Execution_Methods>` called by the user) to evaluate the influence of parameters on performance.

        It also assigns a `ControlSignal` for any `Parameters` of a `Mechanism` `specified for control
        <ParameterPort_Value_Specification>`, and a `ControlProjection` to its correponding `ParameterPort`.

        The ControlMechanism is assigned the `NodeRole` `CONTROLLER`.
        """

        if not isinstance(controller, ControlMechanism):
            raise CompositionError(f"Specification of {repr(CONTROLLER)} arg for {self.name} "
                                   f"must be a {repr(ControlMechanism.__name__)} ")

        # Call with context to avoid recursion by analyze_graph -> _check_initialization_status -> add_controller
        context.source = ContextFlags.METHOD

        # VALIDATE AND ADD CONTROLLER

        # Note:  initialization_status here pertains to controller's own initialization status
        #        (i.e., whether it has been fully instantiated); if not, presumably this is because it is an
        #        OptimizationControlMechanism [OCM] for which the agent_rep has not yet been assigned
        #        (e.g., was constructed in the controller argument of the Composition), in which case assign it here.
        if controller.initialization_status == ContextFlags.DEFERRED_INIT:
            controller._init_args[AGENT_REP] = self
            controller._deferred_init(context=context)

        # Note:  initialization_status here pertains to controller's status w/in the Composition
        #        (i.e., whether any Nodes and/or Projections on which it depends are not yet in the Composition)
        if self._controller_initialization_status != ContextFlags.DEFERRED_INIT:

            # Warn for request to assign the ControlMechanism already assigned and ignore
            if controller is self.controller:
                warnings.warn(f"{controller.name} has already been assigned as the {CONTROLLER} "
                              f"for {self.name}; assignment ignored.")
                return

            # Warn for request to assign ControlMechanism that is already the controller of another Composition
            if hasattr(controller, 'composition') and controller.composition is not self:
                warnings.warn(f"'{controller.name}' has already been assigned as the {CONTROLLER} "
                              f"for '{controller.composition.name}'; assignment to '{self.name}' ignored.")
                return

            # Remove existing controller if there is one
            if self.controller:
                # Warn if current one is being replaced
                warnings.warn(f"The existing {CONTROLLER} for '{self.name}' ('{self.controller.name}') "
                              f"is being replaced by '{controller.name}'.")
                # Remove Projections for old one
                for proj in self.projections.copy():
                    if (proj in self.controller.afferents or proj in self.controller.efferents):
                        self.remove_projection(proj)
                        Projection_Base._delete_projection(proj)
                self.controller.composition=None

        # Assign mutual references between Composition and controller
        controller.composition = self
        self.controller = controller

        # # MODIFIED 12/30/21 NEW:  FIX: THIS IS NOT CORRECT, BECAUSE WITH REGARD TO EXECUTION SEQUENCING,
        # #                                                    CONTROLLER IS NOT THE CONTROLLER OF THE AGENT_REP,
        # #                                                    IT JUST EXECUTES IT.
        # # Deal with agent_rep of controller that is in a nested Composition
        # if (hasattr(self.controller, AGENT_REP)
        #         and self.controller.agent_rep != self
        #         and self.controller.agent_rep in self._get_nested_compositions()):
        #     self.controller.agent_rep.controller = self.controller
        # MODIFIED 12/30/21 END

        # Having controller in nodes is not currently supported (due to special handling of scheduling/execution);
        #    its NodeRole assignment is handled directly by the get_nodes_by_role and get_roles_by_node methods.
        # self._add_node_role(controller, NodeRole.CONTROLLER)

        # Check aux_components relevant to controller
        invalid_aux_components = self._get_invalid_aux_components(controller)
        if invalid_aux_components:
            self._controller_initialization_status = ContextFlags.DEFERRED_INIT
            # Need update here so state_features remains up to date
            self._analyze_graph(context=context)
            return

        # ADD MONITORING COMPONENTS -----------------------------------------------------

        self._handle_allow_probes_for_control(self.controller)

        if self.controller.objective_mechanism:
            # If controller has objective_mechanism, then add it and all associated Projections to Composition
            if self.controller.objective_mechanism not in invalid_aux_components:
                self.controller._validate_monitor_for_control(self._get_all_nodes())
                self.add_node(self.controller.objective_mechanism, required_roles=NodeRole.CONTROLLER_OBJECTIVE)
        else:
            # Otherwise, if controller has any afferent inputs (from items in monitor_for_control), add them
            if self.controller.input_ports and self.controller.input_port.path_afferents:
                self._add_node_aux_components(controller, context)

            # This is set by add_node() automatically if there is an objective_mechanism;
            #    needs to be set here to insure call at run time (to catch any new nodes that may have been added)
            self.needs_update_controller = True

        # Warn if controller is enabled but has no inputs
        if (self.enable_controller
                and not (isinstance(self.controller.input_ports, ContentAddressableList)
                         and self.controller.input_ports
                         and self.controller.afferents)):
            warnings.warn(f"{self.controller.name} for {self.name} is enabled but has no inputs.")

        # ADD MODULATORY COMPONENTS -----------------------------------------------------

        # Get rid of default ControlSignal if it has no ControlProjections
        controller._remove_default_control_signal(type=CONTROL_SIGNAL)
        # Instantiate control specifications locally (on nodes) and/or on controller
        self._instantiate_control_projections(context=context)
        # Instantiate any
        for node in self.nodes:
            self._instantiate_deferred_init_control(node, context)

        # ACTIVATE FOR COMPOSITION -----------------------------------------------------

        self.node_ordering.append(controller)
        self.enable_controller = True
        # FIX: 11/15/21 - SHOULD THIS METHOD BE MOVED HERE (TO COMPOSITION) FROM ControlMechanism
        controller._activate_projections_for_compositions(self)
        self._analyze_graph(context=context)
        if not invalid_aux_components:
            self._controller_initialization_status = ContextFlags.INITIALIZED

    def _instantiate_deferred_init_control(self, node, context=None):
        """
        If node is a Composition with a controller, activate its nodes' deferred init control specs for its controller.
        If it does not have a controller, but self does, activate them for self's controller.

        If node is a Node that has deferred init control specs and self has a controller, activate the deferred init
        control specs for self's controller.

        Called recursively on nodes that are nested Compositions.

        Returns
        -------

        list of hanging control specs that were not able to be assigned for a controller at any level of nesting.

        """
        hanging_control_specs = []
        if node.componentCategory == 'Composition':
            nested_comp = node  # For readability
            for node_in_nested_comp in nested_comp.nodes:
                hanging_control_specs.extend(nested_comp._instantiate_deferred_init_control(node_in_nested_comp,
                                                                                            context=context))
                assert True
        else:
            hanging_control_specs = node._get_parameter_port_deferred_init_control_specs()
        if not self.controller:
            return hanging_control_specs
        else:
            for spec in hanging_control_specs:
                control_signal = self.controller._instantiate_control_signal(control_signal=spec,
                                                                             context=context)
                self.controller.control.append(control_signal)
                self.controller._activate_projections_for_compositions(self)
        return []

    def _get_monitor_for_control_nodes(self):
        """Return dict of {nodes : ControlMechanism that monitors it} for any nodes monitored for control in Composition
        """
        monitored_nodes = {}
        for node in self._all_nodes:
            if isinstance(node, ControlMechanism):
                monitored_nodes.update({spec.owner if isinstance(spec, Port) else spec : node
                                        for spec in node.monitor_for_control})
        return monitored_nodes

    def _get_control_signals_for_composition(self):
        """Return list of ControlSignals specified by Nodes in the Composition

        Generate list of ControlSignal specifications from ParameterPorts of Mechanisms specified for control.
        The specifications can be:
            ControlProjections (with deferred_init())
            # FIX: 9/14/19 - THIS SHOULD ALREADY HAVE BEEN PARSED INTO ControlProjection WITH DEFFERRED_INIT:
            #                OTHERWISE, NEED TO ADD HANDLING OF IT BELOW
            ControlSignals (e.g., in a 2-item tuple specification for the parameter);
            *CONTROL* keyword
            Note:
                The initialization of the ControlProjection and, if specified, the ControlSignal
                are completed in the call to controller_instantiate_control_signal() in add_controller.
        Mechanism can be in the Composition itself, or in a nested Composition that does not have its own controller.
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

    def _get_controller(self, comp=None, context=None):
        """Get controller for which the current Composition is an agent_rep.
        Recursively search enclosing Compositions for controller if self does not have one.
        Use context.composition if there is no controller.
        This is needed for agent_rep that is nested within the Composition to which the controller belongs.
        """
        comp = comp or self
        context = context or Context(source=ContextFlags.COMPOSITION, composition=None)
        if comp.controller:
            return comp.controller
        elif context.composition:
            return context.composition._get_controller(context=context)
        else:
            assert False, f"PROGRAM ERROR: Can't find controller for {comp.name}."

    def reshape_control_signal(self, arr):

        current_shape = np.shape(arr)
        if len(current_shape) > 2:
            newshape = (current_shape[0], current_shape[1])
            newarr = np.reshape(arr, newshape)
            arr = tuple(newarr[i].item() for i in range(len(newarr)))

        return np.array(arr)

    def _instantiate_control_projections(self, context):
        """
        Add any ControlProjections for control specified locally on nodes in Composition
        """

        # Add any ControlSignals specified for ParameterPorts of Nodes already in the Composition
        control_signal_specs = self._get_control_signals_for_composition()
        for ctl_sig_spec in control_signal_specs:
            # FIX: 9/14/19: THIS SHOULD BE HANDLED IN _instantiate_projection_to_port
            #               CALLED FROM _instantiate_control_signal
            #               SHOULD TRAP THAT ERROR AND GENERATE CONTEXT-APPROPRIATE ERROR MESSAGE
            # Don't add any that are already on the ControlMechanism

            # FIX: 9/14/19 - IS THE CONTEXT CORRECT (TRY TRACKING IN SYSTEM TO SEE WHAT CONTEXT IS):
            ctl_signal = self.controller._instantiate_control_signal(control_signal=ctl_sig_spec, context=context)

            self.controller.control.append(ctl_signal)

        # MODIFIED 11/21/21 OLD: FIX: WHY IS THIS INDENTED?  WON'T CALL OUTSIDE LOOP ACTIVATE ALL PROJECTIONS?
            # FIX: 9/15/19 - WHAT IF NODE THAT RECEIVES ControlProjection IS NOT YET IN COMPOSITION:
            #                ?DON'T ASSIGN ControlProjection?
            #                ?JUST DON'T ACTIVATE IT FOR COMPOSITON?
            #                ?PUT IT IN aux_components FOR NODE?
            #                ! TRACE THROUGH _activate_projections_for_compositions TO SEE WHAT IT CURRENTLY DOES
            self.controller._activate_projections_for_compositions(self)

    def _route_control_projection_through_intermediary_pcims(self,
                                                             projection,
                                                             sender,
                                                             sender_mechanism,
                                                             receiver,
                                                             graph_receiver,
                                                             context):
        """
        Takes as input a specification for a projection to a parameter port that is nested n-levels below its sender,
        instantiates and activates ports and projections on intermediary pcims, and returns a new
        projection specification from the original sender to the relevant input port of the pcim of the Composition
        located in the same level of nesting.
        """
        for proj in receiver.mod_afferents:
            if proj.sender.owner == sender_mechanism:
                receiver._remove_projection_to_port(proj)
        for proj in sender.efferents:
            if proj.receiver == receiver:
                sender._remove_projection_from_port(proj)
        modulation = sender.modulation
        interface_input_port = InputPort(owner=graph_receiver.parameter_CIM,
                                         variable=receiver.defaults.value,
                                         reference_value=receiver.defaults.value,
                                         name=PARAMETER_CIM_NAME + "_" + receiver.owner.name + "_" + receiver.name,
                                         context=context)
        graph_receiver.parameter_CIM.add_ports([interface_input_port], context=context)
        # control signal for parameter CIM that will project directly to inner Composition's parameter
        control_signal = ControlSignal(
            modulation=modulation,
            variable=(OWNER_VALUE, functools.partial(graph_receiver.parameter_CIM.get_input_port_position,
                                                     interface_input_port)),
            transfer_function=Identity,
            modulates=receiver,
            name=PARAMETER_CIM_NAME + "_" + receiver.owner.name + "_" + receiver.name,
        )
        if receiver.owner not in graph_receiver.nodes.data + graph_receiver.cims:
            receiver = interface_input_port
        graph_receiver.parameter_CIM.add_ports([control_signal], context=context)
        # add sender and receiver to self.parameter_CIM_ports dict
        for p in control_signal.projections:
            # self.add_projection(p)
            graph_receiver.add_projection(p, receiver=p.receiver, sender=control_signal)
        try:
            sender._remove_projection_to_port(projection)
        except (ValueError, PortError):
            pass
        try:
            receiver._remove_projection_from_port(projection)
        except (ValueError, PortError):
            pass
        receiver = interface_input_port
        return MappingProjection(sender=sender, receiver=receiver)

    def _check_controller_initialization_status(self, context=None):
        """Checks initialization status of controller (if applicable) all Projections or Ports in the Composition
        """

        # Avoid recursion if called from add_controller (by way of analyze_graph) since that is called below
        if context and context.source == ContextFlags.METHOD:
            return

        # If controller is in deferred init, try to instantiate and add it to Composition
        if self.controller and self._controller_initialization_status == ContextFlags.DEFERRED_INIT:
            self.add_controller(self.controller, context=context)

        # Don't bother checking any further if from COMMAND_LINE or COMPOSITION (i.e., anything other than Run)
        #    since no need to detect deferred_init and generate errors until runtime
        if context and context.source in {ContextFlags.COMMAND_LINE, ContextFlags.COMPOSITION}:
            return

        # Check for Mechanisms and Projections in aux_components
        if self._controller_initialization_status == ContextFlags.DEFERRED_INIT:
            invalid_aux_components = self._get_invalid_aux_components(self.controller)
            for component in invalid_aux_components:
                if isinstance(component, Projection):
                    if hasattr(component.receiver, OWNER_MECH):
                        owner = component.receiver.owner_mech
                    else:
                        owner = component.receiver.owner
                    warnings.warn(
                            f"The controller of '{self.name}' has been specified to project to '{owner.name}', "
                            f"but '{owner.name}' is not in '{self.name}' or any of its nested Compositions. "
                            f"This projection will be deactivated until '{owner.name}' is added to' {self.name}' "
                            f"in a compatible way."
                    )
                # FIX: It seems this may never get called, as any specification of a Mechanism in the constructor
                #      for a ControlMechanism automatically instantiates a Projection that triggers the warning above.
                elif isinstance(component, Mechanism):
                    warnings.warn(
                            f"The controller of '{self.name}' has a specification that includes the Mechanism "
                            f"'{component.name}', but '{component.name}' is not in '{self.name}' or any of its "
                            f"nested Compositions. This Mechanism will be deactivated until '{component.name}' is "
                            f"added to '{self.name}' or one of its nested Compositions in a compatible way."
                    )
                    assert False, "WARNING MESSAGE"

        # If Composition is not preparing to execute, allow deferred_inits to persist without warning
        if context and ContextFlags.PREPARING not in context.execution_phase:
            return

        # Check for deferred init ControlProjections
        for node in self.nodes:
            for projection in node.projections:
                if projection.initialization_status == ContextFlags.DEFERRED_INIT:
                    if isinstance(projection, ControlProjection):
                        warnings.warn(f"The '{projection.receiver.name}' parameter of "
                                      f"'{projection.receiver.owner.name}' is specified for control, "
                                      f"but the {COMPOSITION} it is in ('{self.name}') does not have a controller; "
                                      f"if a controller is not added to {self.name} "
                                      f"the control specification will be ignored.")

    def _check_nodes_initialization_status(self, context=None):

        # Avoid recursion if called from add_controller (by way of analyze_graph) since that is called below.
        # Don't bother checking if from COMMAND_LINE or COMPOSITION (i.e., anything other than Run)
        #    since no need to detect deferred_init and generate errors until runtime.
        # If Composition is not preparing to execute, allow deferred_inits to persist without warning
        if context and (context.source in {ContextFlags.METHOD, ContextFlags.COMMAND_LINE, ContextFlags.COMPOSITION}
                        or ContextFlags.PREPARING not in context.execution_phase):
            return

        # NOTE:
        #   May want to add other conditions and warnings here.
        #   Currently just checking for unresolved projections.

        for node in self._partially_added_nodes:
            for proj in self._get_invalid_aux_components(node):
                receiver = proj.receiver.owner
                warnings.warn(
                    f"{node.name} has been specified to project to {receiver.name}, "
                    f"but {receiver.name} is not in {self.name} or any of its nested Compositions. "
                    f"This projection will be deactivated until {receiver.name} is added to {self.name} "
                    f"or a composition nested within it."
                )

    def _get_total_cost_of_control_allocation(self, control_allocation, context, runtime_params):
        total_cost = 0.
        if control_allocation is not None:  # using "is not None" in case the control allocation is 0.

            def get_controller(comp):
                """Get controller for which the current Composition is an agent_rep.
                Recursively search enclosing Compositions for controller if self does not have one.
                Use context.composition to find controller.
                This is needed for agent_rep that is nested within the Composition to which the controller belongs.
                """
                if comp.controller:
                    return comp.controller
                elif context.composition:
                    return get_controller(context.composition)
                else:
                    assert False, f"PROGRAM ERROR: Can't find controller for {self.name}."

            controller = self._get_controller(context=context)

            base_control_allocation = self.reshape_control_signal(controller.parameters.value._get(context))
            candidate_control_allocation = self.reshape_control_signal(control_allocation)

            # Get reconfiguration cost for candidate control signal
            reconfiguration_cost = 0.
            if callable(controller.compute_reconfiguration_cost):
                reconfiguration_cost = controller.compute_reconfiguration_cost([candidate_control_allocation,
                                                                                     base_control_allocation])
                controller.reconfiguration_cost.set(reconfiguration_cost, context)

            # Apply candidate control signal
            controller._apply_control_allocation(candidate_control_allocation,
                                                                context=context,
                                                                runtime_params=runtime_params,
                                                                )

            # Get control signal costs
            other_costs = controller.parameters.costs._get(context) or []
            all_costs = convert_to_np_array(other_costs + [reconfiguration_cost])
            # Compute a total for the candidate control signal(s)
            total_cost = controller.combine_costs(all_costs)

        return total_cost

    # endregion CONTROL

    # ******************************************************************************************************************
    # region ------------------------------------ EXECUTION ------------------------------------------------------------
    # ******************************************************************************************************************

    @handle_external_context()
    def evaluate(
            self,
            predicted_input=None,
            control_allocation=None,
            num_trials=1,
            runtime_params=None,
            base_context=Context(execution_id=None),
            context=None,
            execution_mode:pnlvm.ExecutionMode = pnlvm.ExecutionMode.Python,
            return_results=False,
            block_simulate=False
    ):
        """Run Composition and compute `net_outcomes <ControlMechanism.net_outcome>`

        Runs the `Composition` in simulation mode (i.e., excluding its `controller <Composition.controller>`)
        using the **predicted_input** (state_feature_values and specified **control_allocation** for each run.
        The Composition is run for ***num_trials**.

        If **predicted_input** is not specified, and `block_simulate` is set to True, the `controller
        <Composition.controller>` attempts to use the entire input set provided to the `run <Composition.run>`
        method of the `Composition` as input for the call to `run <Composition.run>`. If it is not, the `controller
        <Composition.controller>` uses the inputs slated for its next or previous execution, depending on whether the
        `controller_mode <Composition.controller_mode>` of the `Composition` is set to `before` or `after`,
        respectively.

       .. note::
            Block simulation can not be used if the Composition's stimuli were specified as a generator.
            If `block_simulate` is set to True and the input type for the Composition was a generator,
            block simulation will be disabled for the current execution of `evaluate <Composition.evaluate>`.

        The `net_outcome <ControlMechanism.net_outcome>` for each run is calculated using the `controller
        <Composition.controller>`'s <ControlMechanism.compute_net_outcome>` function.  Each run is executed
        independently, using the same **predicted_inputs** and **control_allocation**, and a randomly and
        independently sampled seed for the random number generator.  All values are reset to pre-simulation
        values at the end of the simulation.

        Returns the `net_outcome <ControlMechanism.net_outcome>` of a run of the `agent_rep
        <OptimizationControlMechanism.agent_rep>`. If **return_results** is True,
        an array with the results of each run is also returned.
        """

        controller = self._get_controller(context=context)

        # Build input dictionary for simulation
        input_spec = self.parameters.input_specification.get(context)
        if input_spec and block_simulate and not isgenerator(input_spec):
            if isgeneratorfunction(input_spec):
                inputs = input_spec()
            elif isinstance(input_spec, dict):
                inputs = input_spec
        else:
            inputs = predicted_input

        if hasattr(self, '_input_spec') and block_simulate and isgenerator(input_spec):
            warnings.warn(f"The evaluate method of {self.name} is attempting to use block simulation, but the "
                          f"supplied input spec is a generator. Generators can not be used as inputs for block "
                          f"simulation. This evaluation will not use block simulation.")

        # Apply candidate control to signal(s) for the upcoming simulation and determine its cost
        total_cost = self._get_total_cost_of_control_allocation(control_allocation, context, runtime_params)

        # Set up animation for simulation
        # HACK: _animate attribute is set in execute method, but Evaluate can be called on a Composition that has not
        # yet called the execute method, so we need to do a check here too.
        # -DTS
        if not hasattr(self, '_animate'):
            # These are meant to be assigned in run method;  needed here for direct call to execute method
            self._animate = False
        if self._animate is not False and self._animate_simulations is not False:
            animate = self._animate
            buffer_animate_state = None
        else:
            animate = False
            buffer_animate_state = self._animate

        # Run Composition in "SIMULATION" context
        # # MODIFIED 3/28/22 NEW:
        # context.source = ContextFlags.COMPOSITION
        # MODIFIED 3/28/22 END
        context.add_flag(ContextFlags.SIMULATION_MODE)
        context.remove_flag(ContextFlags.CONTROL)

        # EXECUTE run of composition and aggregate results

        # Use reporting options from Report context created in initial (outer) call to run()
        with Report(self, context=context) as report:
            result = self.run(inputs=inputs,
                                    context=context,
                                    runtime_params=runtime_params,
                                    num_trials=num_trials,
                                    animate=animate,
                                    execution_mode=execution_mode,
                                    skip_initialization=True,
                                    )
            context.remove_flag(ContextFlags.SIMULATION_MODE)
            context.execution_phase = ContextFlags.CONTROL
            if buffer_animate_state:
                self._animate = buffer_animate_state

        assert result == self.get_output_values(context)

        # Store simulation results on "base" composition
        if self.initialization_status != ContextFlags.INITIALIZING:
            try:
                self.parameters.simulation_results._get(base_context).append(result)
            except AttributeError:
                self.parameters.simulation_results._set([result], base_context)

        # COMPUTE net_outcome and aggregate in net_outcomes

        # Update input ports in order to get correct value for "outcome" (from objective mech)
        controller._update_input_ports(runtime_params, context)

        # FIX: REFACTOR TO CREATE ARRAY OF INPUT_PORT VALUES FOR OUTCOME_INPUT_PORTS
        outcome_is_array = controller.num_outcome_input_ports > 1
        if not outcome_is_array:
            outcome = controller.input_port.parameters.value._get(context)
        else:
            outcome = []
            for i in range(controller.num_outcome_input_ports):
                outcome.append(controller.parameters.input_ports._get(context)[i].parameters.value._get(context))

        if outcome is None:
            net_outcome = 0.0
        else:
            # Compute net outcome based on the cost of the simulated control allocation (usually, net = outcome - cost)
            net_outcome = controller.compute_net_outcome(outcome, total_cost)

        if return_results:
            return net_outcome, result
        else:
            return net_outcome

    def _infer_target_nodes(self, targets: dict):
        """
        Maps targets onto target mechanisms (as needed by learning)

        Returns
        ---------

        `dict`:
            Dict mapping TargetMechanisms -> target values
        """
        ret = {}
        for node, values in targets.items():
            if (NodeRole.TARGET not in self.get_roles_by_node(node)
                    and NodeRole.LEARNING not in self.get_roles_by_node(node)):
                node_efferent_mechanisms = [x.receiver.owner for x in node.efferents if x in self.projections]
                comparators = [x for x in node_efferent_mechanisms
                               if (isinstance(x, ComparatorMechanism)
                                   and NodeRole.LEARNING in self.get_roles_by_node(x))]
                comparator_afferent_mechanisms = [x.sender.owner for c in comparators for x in c.afferents]
                target_nodes = [t for t in comparator_afferent_mechanisms
                                if (NodeRole.TARGET in self.get_roles_by_node(t)
                                    and NodeRole.LEARNING in self.get_roles_by_node(t))]

                if len(target_nodes) != 1:
                    # Invalid specification: no valid target nodes or ambiguity in which target node to choose
                    raise Exception(f"Unable to infer learning target node from output node {node} of {self.name}")

                ret[target_nodes[0]] = values
            else:
                ret[node] = values
        return ret

    def _parse_learning_spec(self, inputs, targets):
        """
        Converts learning inputs and targets to a standardized form

        Returns
        ---------

        `dict` :
            Dict mapping mechanisms to values (with TargetMechanisms inferred from output nodes if needed)

        `int` :
            Number of input sets in dict for each input node in the Composition
        """

        # Special case for callable inputs
        if callable(inputs):
            return inputs, sys.maxsize

        # 1) Convert from key-value representation of values into separated representation
        if 'targets' in inputs:
            targets = inputs['targets'].copy()

        if 'inputs' in inputs:
            inputs = inputs['inputs'].copy()

        # 2) Convert output node keys -> target node keys (learning always needs target nodes!)
        def _recursive_update(d, u):
            """
            Recursively calls update on dictionaries, which prevents deletion of keys
            """
            for key, val in u.items():
                if isinstance(val, collections.abc.Mapping):
                    d[key] = _recursive_update(d.get(key, {}), val)
                else:
                    d[key] = val
            return d

        if targets is not None:
            targets = self._infer_target_nodes(targets)
            inputs = _recursive_update(inputs, targets)

        # 3) Resize inputs to be of the form [[[]]],
        # where each level corresponds to: <TRIALS <PORTS <INPUTS> > >
        inputs, num_inputs_sets = self._parse_input_dict(inputs)

        return inputs, num_inputs_sets

    def _parse_generator_function(self, inputs):
        """
        Instantiates and parses generator from generator function

        Returns
        -------

        `generator` :
            Generator instance that will be used to yield inputs

        `int` :
            a large int (sys.maxsize), used in place of the number of input sets, since it is impossible to determine
            a priori how many times a generator will yield
        """
        # If a generator function was provided as input, resolve it to a generator and
        # pass to _parse_generator
        _inputs = inputs()
        gen, num_inputs_sets = self._parse_generator(_inputs)
        return gen, num_inputs_sets

    def _parse_generator(self, inputs):
        """
        Returns
        -------
        `generator` :
            Generator instance that will be used to yield inputs

        `int` :
            a large int (sys.maxsize), used in place of the number of input sets, since it is impossible to determine
            a priori how many times a generator will yield
        """
        # It is impossible to determine the number of yields a generator will support, so we return the maximum
        # allowed integer for num_trials here. During execution, we will determine empirically if the generator has
        # yielded to exhaustion by catching StopIteration errors
        num_inputs_sets = sys.maxsize
        return inputs, num_inputs_sets

    def _parse_list(self, inputs):
        """
        Validates that conditions are met to use a list as input, i.e. that there is only one input node. If so, convert
            list to input dict and parse

        Returns
        -------
        `dict` :
            Parsed and standardized input dict

        `int` :
            Number of input sets in dict for each input node in the Composition

        """
        # Lists can only be used as inputs in the case where there is a single input node.
        # Validate that this is true. If so, resolve the list into a dict and parse it.
        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
        if len(input_nodes) == 1:
            _inputs = {next(iter(input_nodes)): inputs}
        else:
            raise CompositionError(
                f"Inputs to {self.name} must be specified in a dictionary with a key for each of its "
                f"{len(input_nodes)} INPUT nodes ({[n.name for n in input_nodes]}).")
        input_dict, num_inputs_sets = self._parse_input_dict(_inputs)
        return input_dict, num_inputs_sets

    def _parse_string(self, inputs):
        """
        Validate that conditions are met to use a string as input, i.e. that there is only one input node and that
        node's default input port has a label matching the provided string. If so, convert the string to an input
        dict and parse.

        Returns
        -------
        `dict` :
            Parsed and standardized input dict

        `int` :
            Number of input sets in dict for each input node in the Composition

        """
        # Strings can only be used as inputs in the case where there is a single input node, and that node's default
        # input port has a label matching the provided string.
        # Validate that this is true. If so, resolve the string into a dict and parse it.
        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
        if len(input_nodes) == 1:
            if inputs in input_nodes[0]._get_standardized_label_dict(INPUT)[0]:
                _inputs = {next(iter(input_nodes)): [inputs]}
        else:
            raise CompositionError(
                f"Inputs to {self.name} must be specified in a dictionary with a key for each of its "
                f"{len(input_nodes)} INPUT nodes ({[n.name for n in input_nodes]}).")
        input_dict, num_inputs_sets = self._parse_input_dict(_inputs)
        return input_dict, num_inputs_sets

    def _parse_function(self, inputs):
        """
        Returns
        -------
        `function` :
            Function that will be used to yield inputs

        `int` :
            Functions must always return 1 trial of input per call, so this always returns 1

        """
        # functions used as inputs must always return a single trial's worth of inputs on each call,
        # so just return the function and 1 as the number of trials
        num_trials = 1
        return inputs, num_trials

    def _validate_single_input(self, receiver, input):
        """Validate a single input for a single receiver.
        If the input is specified without an outer list (i.e. Composition.run(inputs = [1, 1]) instead of
            Composition.run(inputs = [[1], [1]]), add an extra dimension to the input

        Returns
        -------

        `None` or `np.ndarray` or `list` :
            The input, with an added dimension if necessary, if the input is valid. `None` if the input is not valid.
        """

        # Validate that a single input is properly formatted for a receiver.
        _input = []
        if isinstance(receiver, InputPort):
            input_shape = receiver.default_input_shape
        elif isinstance(receiver, Mechanism):
            input_shape = receiver.external_input_shape
        elif isinstance(receiver, Composition):
            input_shape = receiver.input_CIM.external_input_shape
        match_type = self._input_matches_variable(input, input_shape)
        if match_type == 'homogeneous':
            # np.atleast_2d will catch any single-input ports specified without an outer list
            _input = convert_to_np_array(input, 2)
        elif match_type == 'heterogeneous':
            _input = input
        else:
            _input = None
        return _input

    def _parse_input_dict(self, inputs, context=None):
        """
        Validate and parse a dict provided as input to a Composition into a standardized form to be used throughout
            its execution

        Returns
        -------
        `dict` :
            Parsed and standardized input dict

        `int` :
            Number of input sets (i.e., trials' worths of inputs) in dict for each input node in the Composition

        """
        # parse a user-provided input dict to format it properly for execution.
        # compute number of input sets and return that as well
        _inputs = self._parse_names_in_inputs(inputs)
        _inputs = self._parse_labels(_inputs)
        self._validate_input_dict_keys(_inputs)
        _inputs = self._instantiate_input_dict(_inputs)
        _inputs = self._flatten_nested_dicts(_inputs)
        _inputs = self._validate_input_shapes(_inputs)
        num_inputs_sets = len(next(iter(_inputs.values()),[]))
        return _inputs, num_inputs_sets

    def _parse_names_in_inputs(self, inputs):
        names = []
        # Get keys that are names rather than Components
        for key in inputs:
            if isinstance(key, str):
                names.append(key)
        # named_entries = [(node, node.name) for node in self.get_nodes_by_role(NodeRole.INPUT) if node.name in names]
        named_entries = []
        for node in self.get_nodes_by_role(NodeRole.INPUT):
            if node.name in names:
                named_entries.append((node, node.name))
            else:
                for port in node.input_ports:
                    if port.full_name in names:
                        named_entries.append((port, port.full_name))
        # Replace name with node itself in key
        for node, name in named_entries:
            inputs[node] = inputs.pop(name)
        return inputs

    def _parse_labels(self, inputs, mech=None, port=None, context=None):
        """
        Traverse input dict and resolve any input or output labels to their numeric values
        If **port** is passed, inputs is only for a single port, so manage accordingly

        Returns
        -------

        `dict` :
            The input dict, with inputs with labels replaced by corresponding numeric values
        """

        # the nested list comp below is necessary to retrieve target nodes of learning pathways,
        # because the PathwayRole enum is not importable into this module
        target_to_output = {path.target: path.output for path in self.pathways
                            if 'LEARNING' in [role.name for role in path.roles]}
        if mech:
            target_nodes_of_learning_pathways = [path.target if path.learning_components else None
                                                 for path in self.pathways]
            label_type = INPUT if mech not in target_nodes_of_learning_pathways else OUTPUT
            label_mech = mech if mech not in target_to_output else target_to_output[mech]
            labels = label_mech._get_standardized_label_dict(label_type)
        if type(inputs) == dict:
            _inputs = {}
            for k,v in inputs.items():
                if isinstance(k, Mechanism) and \
                   (target_to_output[k].output_labels_dict if k in target_to_output else k.input_labels_dict):
                    _inputs.update({k:self._parse_labels(v, k)})  # Full node's worth of inputs, so don't pass port
                else:
                    # Call _parse_labels for any Nodes with input_labels_dicts in nested Composition(s)
                    if (isinstance(k, Composition)
                            and any(n.input_labels_dict
                                    for n in k._get_nested_nodes_with_same_roles_at_all_levels(k,NodeRole.INPUT))):
                        if nesting_depth(v) == 2:
                            # Enforce that even single trial specs user outer trial dimension (for consistency below)
                            v = [v]
                        for t in range(len(v)):
                            for i, cim_port in enumerate(k.input_CIM.input_ports):
                                port, mech_with_labels, __ = k.input_CIM._get_destination_info_from_input_CIM(cim_port)
                                # Get only a port's worth of input, so signify by passing port with input,
                                #   which is also need since it is not bound to owning Mechanism in input_CIM,
                                #   so its index can't be determined in recursive call to _parse_labels below
                                v[t][i] = k._parse_labels(v[t][i],mech_with_labels, port)
                        _inputs.update({k:v})
                    else:
                        _inputs.update({k:v})
        elif type(inputs) == list or type(inputs) == np.ndarray:
            _inputs = []
            for i in range(len(inputs)): # One for each port if full node's worth, else inputs = input for port
                if port:
                    # port passed in, since it is not bound to owner in input_CIM, so index can't be determined locally
                    # also signifies input is to be treated as just for that port (not entire node), so 1 dim less
                    port_index = mech.input_ports.index(port)
                    if not isinstance(inputs[0], str):
                        # If input for port is not a string, no further processing need, so just return as is
                        _inputs = inputs
                        break
                else:
                    # No port passed in, so use primary InputPort if only one input, or i if inputs for multiple ports
                    port_index = 0 if len(labels) == 1 else i
                stimulus = inputs[i] # input for port
                if type(stimulus) == list or type(stimulus) == np.ndarray:
                    _inputs.append(self._parse_labels(inputs[i], mech))
                elif type(stimulus) == str:
                    if not labels:
                        raise CompositionError(f"Inappropriate use of str ({repr(stimulus)}) as a stimulus for "
                                               f"{mech.name} in {self.name}: it does not have an input_labels_dict.")
                    try:
                        if len(inputs) == 1:
                            _inputs = np.atleast_1d(labels[port_index][stimulus])
                        else:
                            _inputs.append(labels[port_index][stimulus])
                    except KeyError as e:
                        raise CompositionError(f"Inappropriate use of {repr(stimulus)} as a stimulus for {mech.name} "
                                               f"in {self.name}: it is not a label in its input_labels_dict.")
                else:
                    _inputs.append(stimulus)

        return _inputs

    def _validate_input_dict_keys(self, inputs):
        """Validate that keys of inputs are all legal:
            - they are all InputPorts, Mechanisms or Compositions;
            - they are all (or InputPorts of) INPUT Nodes of Composition at any level of nesting;
            - an InputPort and the Mechanism to which it belongs are not *both* specified;
            - an InputPort of an input_CIM and the Composition to which it belongs are not *both* specified;
            - an InputPort or Mechanism and any Composition under which it is nested are not *both* specified.
        """

        # Validate that keys for inputs are all legal *types*
        bad_entries = [key for key in inputs if not isinstance(key, (InputPort, Mechanism, Composition))]
        if bad_entries:
            bad_entry_names = [repr(key.full_name) if isinstance(key, Port) else repr(key.name) for key in bad_entries]
            raise RunError(f"The following items specified in the 'inputs' arg of the run() method for "
                           f"'{self.name}' that are not a Mechanism, Composition, or an InputPort of one: "
                           f"{', '.join(bad_entry_names)}.")

        # Validate that keys for inputs all are or belong to *INPUT Nodes* of Composition (at any level of nesting)
        all_allowable_entries = self._get_input_receivers(type=PORT) \
                                + self._get_input_receivers(type=NODE, comp_as_node=ALL)
        bad_entries = [key for key in inputs if key not in all_allowable_entries]
        if bad_entries:
            bad_entry_names = [repr(key.full_name) if isinstance(key, Port) else repr(key.name) for key in bad_entries]
            raise RunError(f"The following items specified in the 'inputs' arg of the run() method for '{self.name}' "
                           f"are not INPUT Nodes of that Composition (nor InputPorts of them): "
                           f"{', '.join(bad_entry_names)}.")

        # Validate that an InputPort *and* its owner are not *both* specified in inputs
        bad_entries = [key.full_name for key in inputs if isinstance(key, InputPort) and key.owner in inputs]
        if bad_entries:
            raise RunError(f"The 'inputs' arg of the run() method for '{self.name}' includes specifications of the "
                           f"following InputPorts *and* the Mechanisms to which they belong; only one or the other "
                           f"can be specified as inputs to run():  {', '.join(bad_entries)}.")

        # Validate that an InputPort of an input_CIM *and* its Composition are not *both* specified in inputs
        #    (this is unlikely but possible)
        bad_entries = [key.full_name for key in inputs
                       if (isinstance(key, InputPort)
                           and isinstance(key.owner, CompositionInterfaceMechanism)
                           and key.owner.composition in inputs)]
        if bad_entries:
            raise RunError(f"The 'inputs' arg of the run() method for '{self.name}' includes specifications of the "
                           f"following InputPort(s) of a CompositionInterfaceMechanism *and* the Composition to which "
                           f"they belong; only one or the other can be specified as inputs to run(): "
                           f"{', '.join(bad_entries)}.")

        # # Validate that InputPort or Mechanism and the Composition(s) under which it is nested are not both specified
        def check_for_items_in_nested_comp(comp):
            bad_entries = []
            for node in comp._all_nodes:  # note: this only includes nodes as top level of comp
                if isinstance(node, Composition) and node in inputs:
                    all_nested_items = node._get_input_receivers(type=PORT) \
                                       + node._get_input_receivers(type=NODE, comp_as_node=ALL)
                    bad_entries.extend([(entry, node) for entry in inputs if entry in all_nested_items])
                    bad_entries.extend(check_for_items_in_nested_comp(node))
            return bad_entries
        bad_entries = check_for_items_in_nested_comp(self)
        if bad_entries:
            bad_entry_names = [(key.full_name, comp.name) if isinstance(key, Port) else (key.name, comp.name)
                               for key,comp in bad_entries]
            raise RunError(f"The 'inputs' arg of the run() method for '{self.name}' includes specifications of the "
                           f"following InputPorts or Mechanisms *and* the Composition within which they are nested: "
                           # f"{', '.join(bad_entry_names)}.")
                           f"{bad_entry_names}.")

    def _instantiate_input_dict(self, inputs):
        """Implement dict with all INPUT Node of Composition as keys and their assigned inputs or defaults as values
        **inputs** can contain specifications for inputs to InputPorts, Mechanisms and/or nested Compositions,
            that can be at any level of nesting within self.
        Consolidate any entries of **inputs** with InputPorts as keys to Mechanism or Composition entries
        If any INPUT Nodes of Composition are not included, add them to the input_dict using their default values.
        InputPort entries must specify either a single trial or the same number as all other InPorts for that Node:
          - preprocess InputPorts for a Node to determine maximum number of trials specified, and use to set mech_shape
          - if more than one trial is specified for any InputPort, assign fillers to ones that specify only one trial
          (this does not apply to Mechanism or Composition specifications, as they are tested in validate_input_shapes)

        Return input_dict, with added entries for any INPUT Nodes or InputPorts for which input was not provided
        """

        input_dict = {}
        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
        remaining_inputs = set(inputs)

        # Construct input_dict from input_nodes of self
        for INPUT_Node in input_nodes:

            # If entry is for an INPUT_Node of self, assign the entry directly to input_dict and proceed to next
            if INPUT_Node in inputs:
                input_dict[INPUT_Node] = inputs[INPUT_Node]
                remaining_inputs.remove(INPUT_Node)
                continue

            # If INPUT_Node is a Composition, get its input_CIM as mech
            mech = INPUT_Node if isinstance(INPUT_Node, Mechanism) else INPUT_Node.input_CIM
            INPUT_input_ports = mech.input_ports
            input_port_entries = {}
            inputs_to_remove = set()

            # Look for entries of inputs dict with keys that are input_ports of mech, or Composition's input_CIM
            #    Note: need to cycle through all inputs before constructing entries for INPUT Node
            #          since there may be multiple Port entries for a given INPUT Node that may differ in their specs,
            #          so need to process all of them to determine proper shape for input
            for input_recvr in remaining_inputs:

                # Get input spec and standardize port spec as 2d to deal with any specs in time series format
                port_inputs = np.atleast_2d(inputs[input_recvr])

                # Entry is InputPort of INPUT_Node itself (usually of a standard Mechanism, but could be of input_CIM)
                if input_recvr in mech.input_ports:
                    input_port_entries[input_recvr] = port_inputs
                    inputs_to_remove.add(input_recvr)

                # If INPUT_Node is a Composition, check if input_recvr is for an INPUT Node nested under it;
                #    if so, get the input_port(s) of mech.input_CIM to which its inputs correspond
                elif (isinstance(INPUT_Node, Composition)
                      and ((input_recvr.owner if isinstance(input_recvr, Port) else input_recvr) in
                           INPUT_Node._get_nested_nodes_with_same_roles_at_all_levels(INPUT_Node, NodeRole.INPUT))):
                    if isinstance(input_recvr, Port):
                        # Spec is for Port so assign to entry for corresponding input_CIM_input_port of INPUT_Node
                        input_ports = [input_recvr]
                    else:
                        # Spec is for Mechanism or Composition so get its input_ports
                        input_ports = (input_recvr.input_ports if isinstance(input_recvr, Mechanism)
                                       else input_recvr.input_CIM.input_ports)
                    # For each port in input_ports of Mech or Compositions input_CIM
                    for i, input_port in enumerate(input_ports):
                        # Get corresponding InputPort of input_CIM for INPUT_Node
                        input_CIM_input_port, _ = self._get_external_cim_input_port(input_port, self)
                        assert input_CIM_input_port.owner == mech, \
                            f"PROGRAM ERROR: Unexpected input_CIM_input_port retrieved for entry ({input_recvr}) " \
                            f"in inputs to '{self.name}'."
                        # Assign spec for InputPort to entry for corresponding InputPort on input_CIM of INPUT_Node
                        input_port_entries[input_CIM_input_port] = port_inputs[i]
                    inputs_to_remove.add(input_recvr)

            # Get max number of trials across specified input_ports of INPUT_Node
            max_num_trials = 1
            for port in input_port_entries:
                assert mech == port.owner
                port_input = input_port_entries[port]
                # Get number of trials of input specified for Port
                num_trials = len(port_input)
                if max_num_trials != 1 and num_trials not in {1, max_num_trials}:
                    raise CompositionError(f"Number of trials of input specified for {port.full_name} of {node.name} "
                                           f"({num_trials}) is different from the number ({max_num_trials}) "
                                           f"specified for one or more others.")
                max_num_trials = max(num_trials, max_num_trials)

            # Construct node_input_shape based on max_num_trials across all input_ports for mech
            # - shape as 3d by adding outer dim = max_num trials to accommodate potential trial-series input
            node_input = np.empty(tuple([max_num_trials] +
                                        list(np.array(mech.external_input_shape).shape)),
                                  dtype='object').tolist()
            # - move ports to outer access for processing below
            node_input = np.swapaxes(np.atleast_3d(np.array(node_input, dtype=object)),0,1).tolist()

            # Assign specs to ports of INPUT_Node, using ones in input_port_entries or defaults
            for i, port in enumerate(INPUT_input_ports):
                if port in input_port_entries:
                    # Assume input is for all trials
                    port_spec = np.atleast_2d(input_port_entries[port]).tolist()
                    if len(port_spec) < max_num_trials:
                        # If input is not for all trials, ensure that it is only for a single trial
                        assert len(port_spec) == 1, f"PROGRAM ERROR: Length of port_spec for '{port.full_name}' " \
                                                    f"in input to '{self.name}' ({len(port_spec)}) should now be " \
                                                    f"1 or {max_num_trials}."
                        # Assign the input for the single trial over all trials
                        port_spec = [np.array(port_spec[0]).tolist()] * max_num_trials
                else:
                    # Assign default input to Port for all trials
                    port_spec = [np.array(port.default_input_shape).tolist()] * max_num_trials
                node_input[i] = port_spec

            # Put trials back in outer axis
            input_dict[INPUT_Node] = np.swapaxes(np.atleast_3d(np.array(node_input, dtype=object)),0,1).tolist()
            remaining_inputs = remaining_inputs - inputs_to_remove

        if remaining_inputs:
            assert False, f"PROGRAM ERROR: the following items specified in the 'inputs' arg of the run() method " \
                          f"for '{self.name}' are not INPUT Nodes of that Composition (nor InputPorts of them): " \
                          f"{remaining_inputs} -- SHOULD HAVE RAISED ERROR IN composition._validate_input_dict_keys()"

        # If any INPUT Nodes of the Composition are not specified, add them and assign default_external_input_values
        for node in input_nodes:
            if node not in input_dict:
                input_dict[node] = node.external_input_shape

        return input_dict

    def _flatten_nested_dicts(self, inputs):
        """
        Converts inputs provided in the form of a dict for a nested Composition to a list corresponding to the
            Composition's input CIM ports

        Returns
        -------

        `dict` :
            The input dict, with nested dicts corresponding to nested Compositions converted to lists

        """
        # Inputs provided for nested compositions in the form of a nested dict need to be converted into a list,
        # to be provided to the outer Composition's input port that corresponds to the nested Composition
        _inputs = {}
        for node, inp in inputs.items():
            if node.componentType == 'Composition' and type(inp) == dict:
                # If there are multiple levels of nested dicts, we need to convert them starting from the deepest level,
                # so recurse down the chain here
                inp, num_trials = node._parse_input_dict(inp)
                translated_stimulus_dict = {}

                # first time through the stimulus dictionary, assemble a dictionary in which the keys are input CIM
                # InputPorts and the values are lists containing the first input value
                for nested_input_node, values in inp.items():
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
                _inputs[node] = adjusted_stimulus_list
            else:
                _inputs.update({node:inp})
        return _inputs

    def _validate_input_shapes(self, inputs):
        """
        Validates that all inputs provided in input dict are valid

        Returns
        -------

        `dict` :
            The input dict, with shapes corrected if necessary.

        """
         # Loop over all dictionary entries to validate their content and adjust any convenience notations:

         # (1) Replace any user provided convenience notations with values that match the following specs:
         # a - all dictionary values are lists containing an input value for each trial (even if only one trial)
         # b - each input value is a 2d array that matches variable
         # example: { Mech1: [Fully_specified_input_for_mech1_on_trial_1, Fully_specified_input_for_mech1_on_trial_2 … ],
         #            Mech2: [Fully_specified_input_for_mech2_on_trial_1, Fully_specified_input_for_mech2_on_trial_2 … ]}
         # (2) Verify that all nodes provide the same number of inputs (check length of each dictionary value)
        _inputs = {}
        input_lengths = set()
        inputs_to_duplicate = []
        # loop through input dict
        for receiver, stimulus in inputs.items():
            # see if the entire stimulus set provided is a valid input for the receiver (i.e. in the case of a call with a
            # single trial of provided input)
            _input = self._validate_single_input(receiver, stimulus)
            if _input is not None:
                _input = [_input]
            else:
                # if _input is None, it may mean there are multiple trials of input in the stimulus set,
                #     so in list comprehension below loop through and validate each individual input;
                _input = [self._validate_single_input(receiver, single_trial_input) for single_trial_input in stimulus]
                # Look for any bad ones (for which _validate_single_input() returned None) and report if found
                if any(i is None for i in _input):
                    if isinstance(receiver, InputPort):
                        receiver_shape = receiver.default_input_shape
                        receiver_name = receiver.full_name
                    elif isinstance(receiver, Mechanism):
                        receiver_shape = receiver.external_input_shape
                        receiver_name = receiver.name
                    elif isinstance(receiver, Composition):
                        receiver_shape = receiver.input_CIM.external_input_shape
                        receiver_name = receiver.name
                    # # MODIFIED 3/12/22 OLD:
                    # bad_stimulus = np.atleast_1d(np.squeeze(np.array(stimulus[_input.index(None)], dtype=object)))
                    # correct_stimulus = np.atleast_1d(np.array(receiver_shape[_input.index(None)], dtype=object))
                    # err_msg = f"Input stimulus ({bad_stimulus}) for {receiver_name} is incompatible with " \
                    #           f"the shape of its external input ({correct_stimulus})."
                    # MODIFIED 3/12/22 NEW:
                    # FIX: MIS-REPORTS INCOMPATIBLITY AS BEING FOR SHAPE IF NUM TRIALS IS DIFFERENT FOR DIFF PORTS
                    #      SHOULD BE HANDLED SAME AS FOR DIFFERNCE ACROSS NODES (PER BELOW)
                    receiver_shape = np.atleast_1d(np.squeeze(np.array(receiver_shape, dtype=object)))
                    bad_stimulus = [stim for stim, _inp in zip(stimulus, _input) if _inp is None]
                    bad_stimulus = np.atleast_1d(np.squeeze(np.array(bad_stimulus, dtype=object)))
                    err_msg = f"Input stimulus ({bad_stimulus}) for {receiver_name} is incompatible with " \
                              f"the shape of its external input ({receiver_shape})."
                    # MODIFIED 3/12/22 END
                    # 8/3/17 CW: I admit the error message implementation here is very hacky;
                    # but it's at least not a hack for "functionality" but rather a hack for user clarity
                    if "KWTA" in str(type(receiver)):
                        err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros " \
                                            "(or other values) to represent the outside stimulus for " \
                                            "the inhibition InputPort, and for Compositions, put your inputs"
                    raise RunError(err_msg)
            _inputs[receiver] = _input
            input_length = len(_input)
            if input_length == 1:
                inputs_to_duplicate.append(receiver)
            # track input lengths. stimulus sets of length 1 can be duplicated to match another stimulus set length.
            # there can be at maximum 1 other stimulus set length besides 1.
            input_lengths.add(input_length)
        if 1 in input_lengths:
            input_lengths.remove(1)
        if len(input_lengths) > 1:
            raise CompositionError(f"The input dictionary for {self.name} contains input specifications of different "
                                    f"lengths ({input_lengths}). The same number of inputs must be provided for each "
                                   f"receiver in a Composition.")
        elif len(input_lengths) > 0:
            num_trials = list(input_lengths)[0]
            for mechanism in inputs_to_duplicate:
                # hacky, but need to convert to list to use * syntax to duplicate element
                if type(_inputs[mechanism]) == np.ndarray:
                    _inputs[mechanism] = _inputs[mechanism].tolist()
                _inputs[mechanism] *= num_trials
        return _inputs

    def _parse_run_inputs(self, inputs, context=None):
        """
        Takes user-provided input for entire run and parses it according to its type

        Returns
        -------

        Parsed input dict

        The number of inputs sets included in the input
        """
        # handle user-provided input based on input type. return processd inputs and num_inputs_sets
        if not inputs:
            _inputs, num_inputs_sets = self._parse_input_dict({})
        elif isgeneratorfunction(inputs):
            _inputs, num_inputs_sets = self._parse_generator_function(inputs)
        elif isgenerator(inputs):
            _inputs, num_inputs_sets = self._parse_generator(inputs)
        elif callable(inputs):
            _inputs, num_inputs_sets = self._parse_function(inputs)
        elif type(inputs) == list:
            _inputs, num_inputs_sets = self._parse_list(inputs)
        elif type(inputs) == dict:
            _inputs, num_inputs_sets = self._parse_input_dict(inputs)
        elif type(inputs) == str:
            _inputs, num_inputs_sets = self._parse_string(inputs)
        else:
            raise CompositionError(
                f"Provided inputs {inputs} is in a disallowed format. Inputs must be provided in the form of "
                f"a dict, list, function, or generator. "
                f"See https://princetonuniversity.github.io/PsyNeuLink/Composition.html#composition-run "
                f"for details and formatting instructions for each input type."
            )
        return _inputs, num_inputs_sets

    def _parse_trial_inputs(self, inputs, trial_num):
        """
        Extracts inputs for a single trial and parses it in accordance with its type

        Returns
        -------

        `dict` :
            Input dict parsed for a single trial of a Composition's execution

        """
        # parse and return a single trial's worth of inputs.
        # this method is intended to run BEFORE a call to Composition.execute
        if callable(inputs):
            try:
                inputs, _ = self._parse_input_dict(inputs(trial_num))
                i = 0
            except TypeError as e:
                error_text = e.args[0]
                if f" takes 0 positional arguments but 1 was given" in error_text:
                    raise CompositionError(f"{error_text}: requires arg for trial number")
                else:
                    raise CompositionError(f"Problem with function provided to 'inputs' arg of {self.name}.run")
        elif isgenerator(inputs):
            inputs, _ = self._parse_input_dict(inputs.__next__())
            i = 0
        else:
            num_inputs_sets = len(next(iter(inputs.values())))
            i = trial_num % num_inputs_sets
        next_inputs = {node:inp[i] for node, inp in inputs.items()}
        return next_inputs

    def _validate_execution_inputs(self, inputs):
        """
        Validates and returns the formatted input dict for a single execution

        Returns
        -------

        `dict` :
            Input dict parsed for a single trial of a Composition's execution

        """
        # validate a single execution's worth of inputs
        # this method is intended to run DURING a call to Composition.execute
        _inputs = {}
        for node, inp in inputs.items():
            if isinstance(node, Composition) and type(inp) == dict:
                inp = node._parse_input_dict(inp)
            if np.array(inp).ndim == 3:
                # If inp formatted for trial series, get only one one trial's worth of inputs to test
                inp = np.squeeze(inp, 0)
            inp = self._validate_single_input(node, inp)
            if inp is None:
                raise CompositionError(f"Input stimulus ({inp}) for {node.name} is incompatible "
                                       f"with its variable ({node.external_input_shape}).")
            _inputs[node] = inp
        return _inputs

    # ******************************************************************************************************************
    #                                           EXECUTION
    # ******************************************************************************************************************

    # MODIFIED 3/28/22 OLD:
    @handle_external_context()
    # # MODIFIED 3/28/22 NEW:
    # @handle_external_context(source = ContextFlags.COMMAND_LINE)
    # MODIFIED 3/28/22 END
    def run(
            self,
            inputs=None,
            num_trials=None,
            initialize_cycle_values=None,
            reset_stateful_functions_to=None,
            reset_stateful_functions_when=Never(),
            skip_initialization=False,
            clamp_input=SOFT_CLAMP,
            runtime_params=None,
            call_before_time_step=None,
            call_after_time_step=None,
            call_before_pass=None,
            call_after_pass=None,
            call_before_trial=None,
            call_after_trial=None,
            termination_processing=None,
            skip_analyze_graph=False,
            report_output:ReportOutput=ReportOutput.OFF,
            report_params:ReportParams=ReportParams.OFF,
            report_progress=ReportProgress.OFF,
            report_simulations=ReportSimulations.OFF,
            report_to_devices=None,
            animate=False,
            log=False,
            scheduler=None,
            scheduling_mode: typing.Optional[SchedulingMode] = None,
            execution_mode:pnlvm.ExecutionMode = pnlvm.ExecutionMode.Python,
            default_absolute_time_unit: typing.Optional[pint.Quantity] = None,
            context=None,
            base_context=Context(execution_id=None),
            ):
        """Pass inputs to Composition, then execute sets of nodes that are eligible to run until termination
        conditions are met.

        See `Composition_Execution` for details of formatting input specifications.\n
        Use `get_input_format <Composition.get_input_format>` method to see example of input format.\n
        Use **animate** to generate a gif of the execution sequence.

        Arguments
        ---------

        inputs: Dict{`INPUT` `Node <Composition_Nodes>` : list}, function or generator : default None specifies
            the inputs to each `INPUT` `Node <Composition_Nodes>` of the Composition in each `TRIAL <TimeScale.TRIAL>`
            executed during the run (see `Composition_Execution_Inputs` for additional information about format, and
            `get_input_format <Composition.get_input_format>` method for generating an example of the input format for
            the Composition). If **inputs** is not specified, the `default_variable <Component_Variable>` for
            each `INPUT` Node is used as its input on `TRIAL <TimeScale.TRIAL>`.

        num_trials : int : default 1
            typically, the composition will infer the number of trials from the length of its input specification.
            To reuse the same inputs across many trials, an input dictionary can be specified with lists of length 1,
            or use default inputs, and select a number of trials with num_trials.

        initialize_cycle_values : Dict { Node: Node Value } : default None
            sets the value of specified `Nodes <Composition_Nodes>` before the start of the run.  All specified
            Nodes must be in a `cycle <Composition_Graph>` (i.e., designated with with `NodeRole` `CYCLE
            <NodeRole.CYCLE>`; otherwise, a warning is issued and the specification is ignored). If a Node in
            a cycle is not specified, it is assigned its `default values <Parameter_Defaults>` when initialized
            (see `Composition_Cycles_and_Feedback` additional details).

        reset_stateful_functions_to : Dict { Node : Object | iterable [Object] } : default None
            object or iterable of objects to be passed as arguments to nodes' reset methods when their
            respective reset_stateful_function_when conditions are met. These are used to seed the stateful attributes
            of Mechanisms that have stateful functions. If a node's reset_stateful_function_when condition is set to
            Never, but they are listed in the reset_stateful_functions_to dict, then they will be reset once at the
            beginning of the run, using the provided values. For a more in depth explanation of this argument, see
            `Resetting Parameters of stateful <Composition_Reset>`.

        reset_stateful_functions_when :  Dict { Node: Condition } | Condition : default Never()
            if type is dict, sets the reset_stateful_function_when attribute for each key Node to its corresponding value
            Condition. if type is Condition, sets the reset_stateful_function_when attribute for all nodes in the
            Composition that currently have their reset_stateful_function_when conditions set to `Never <Never>`.
            in either case, the specified Conditions persist only for the duration of the run, after which the nodes'
            reset_stateful_functions_when attributes are returned to their previous Conditions. For a more in depth
            explanation of this argument, see `Resetting Parameters of stateful <Composition_Reset>`.

        skip_initialization : bool : default False

        clamp_input : enum.Enum[SOFT_CLAMP|HARD_CLAMP|PULSE_CLAMP|NO_CLAMP] : default SOFT_CLAMP
            specifies how inputs are handled for the Composition's `INPUT` `Nodes <Composition_Nodes>`.

            COMMENT:
               BETTER DESCRIPTION NEEDED
            COMMENT

        runtime_params : Dict[Node: Dict[Parameter: Tuple(Value, Condition)]] : default None
            nested dictionary of (value, `Condition`) tuples for parameters of Nodes (`Mechanisms <Mechanism>` or
            `Compositions <Composition>` of the Composition; specifies alternate parameter values to be used only
            during this `RUN` when the specified `Condition` is met (see `Composition_Runtime_Params` for
            additional informaton).

        call_before_time_step : callable  : default None
            specifies fuction to call before each `TIME_STEP` is executed.

        call_after_time_step : callable  : default None
            specifies fuction to call after each `TIME_STEP` is executed.

        call_before_pass : callable  : default None
            specifies fuction to call before each `PASS` is executed.

        call_after_pass : callable  : default None
            specifies fuction to call after each `PASS` is executed.

        call_before_trial : callable  : default None
            specifies fuction to call before each `TRIAL <TimeScale.TRIAL>` is executed.

        call_after_trial : callable  : default None
            specifies fuction to call after each `TRIAL <TimeScale.TRIAL>` is executed.

        termination_processing : Condition  : default None
            specifies
            `termination Conditions <Scheduler_Termination_Conditions>`
            to be used for the current `RUN <TimeScale.RUN>`. To change
            these conditions for all future runs, use
            `Composition.termination_processing` (or
            `Scheduler.termination_conds`)

        skip_analyze_graph : bool : default False
            setting to True suppresses call to _analyze_graph()

            COMMENT:
               BETTER DESCRIPTION NEEDED
            COMMENT

        report_output : ReportOutput : default ReportOutput.OFF
            specifies whether to show output of the Composition and its `Nodes <Composition_Nodes>` trial-by-trial as
            it is generated; see `Report_Output` for additional details and `ReportOutput` for options.

        report_params : ReportParams : default ReportParams.OFF
            specifies whether to show values the `Parameters` of the Composition and its `Nodes <Composition_Nodes>`
            as part of the output report; see `Report_Output` for additional details and `ReportParams` for options.

        report_progress : ReportProgress : default ReportProgress.OFF
            specifies whether to report progress of execution in real time; see `Report_Progress` for additional
            details.

        report_simulations : ReportSimulations : default ReportSimulatons.OFF
            specifies whether to show output and/or progress for `simulations <OptimizationControlMechanism_Execution>`
            executed by the Composition's `controller <Composition_Controller>`; see `Report_Simulations` for
            additional details.

        report_to_devices : list(ReportDevices) : default ReportDevices.CONSOLE
            specifies where output and progress should be reported; see `Report_To_Devices` for additional
            details and `ReportDevices` for options.

        animate : dict or bool : default False
            specifies use of the `show_graph`show_graph <ShowGraph.show_graph>` method to generate
            a gif movie showing the sequence of Components executed in a run (see `example
            <BasicsAndPrimer_Stroop_Example_Animation_Figure>`). A dict can be specified containing
            options to pass to the `show_graph <ShowGraph.show_graph>` method; each key must be a legal
            argument for the `show_graph <ShowGraph.show_graph>` method, and its value a specification for that
            argument.  The entries listed below can also be included in the dict to specify parameters of the
            animation.  If the **animate** argument is specified simply as `True`, defaults are used for all
            arguments of `show_graph <ShowGraph.show_graph>` and the options below:

            * *UNIT*: *EXECUTION_SET* or *COMPONENT* (default=\\ *EXECUTION_SET*\\ ) -- specifies which Components
              to treat as active in each call to `show_graph() <ShowGraph.show_graph>`. *COMPONENT* generates an
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

            * *MOVIE_NAME*: str (default=\\ `name <Composition.name>` + 'movie') -- specifies the name to be used
              for the movie file; it is automatically appended with '.gif'.

            * *SAVE_IMAGES*: bool (default=\\ `False`\\ ) -- specifies whether to save each of the images used to
              construct the animation in separate gif files, in addition to the file containing the animation.

            * *SHOW*: bool (default=\\ `False`\\ ) -- specifies whether to show the animation after it is
              constructed, using the OS's default viewer.

        log : bool or LogCondition : default False
            Sets the `log_condition <Parameter.log_condition>` for every primary `node <Composition.nodes>` and
            `projection <Composition.projections>` in the Composition, if it is not already set.

            .. note::
               as when setting the `log_condition <Parameter.log_condition>` directly, a value of `True` will
               correspond to the `EXECUTION` `LogCondition <LogCondition.EXECUTION>`.

        scheduler : Scheduler : default None
            the scheduler object that owns the conditions that will instruct the execution of the Composition.
            If not specified, the Composition will use its automatically generated scheduler.

        scheduling_mode : SchedulingMode[STANDARD|EXACT_TIME] : default None
            if specified, sets the `scheduling mode <SchedulingMode>`
            for the current and all future runs of the Composition. See
            `Scheduler_Execution`

        execution_mode : enum.Enum[Auto|LLVM|LLVMexec|LLVMRun|Python|PTXExec|PTXRun] : default Python
            specifies whether to run using the Python interpreter or a `compiled mode <Composition_Compilation>`.
            False is the same as ``Python``;  True tries LLVM compilation modes, in order of power, progressively
            reverting to less powerful modes (in the order of the options listed), and to Python if no compilation
            mode succeeds (see `Composition_Compilation` for explanation of modes). PTX modes are used for
            CUDA compilation.

        default_absolute_time_unit : ``pint.Quantity`` : ``1ms``
            if not otherwise determined by any absolute **conditions**,
            specifies the absolute duration of a `TIME_STEP`. See
            `Scheduler.default_absolute_time_unit`

        context : `execution_id <Context.execution_id>` : default `default_execution_id`
            context in which the `Composition` will be executed;  set to self.default_execution_id ifunspecified.

        base_context : `execution_id <Context.execution_id>` : Context(execution_id=None)
            the context corresponding to the execution context from which this execution will be initialized,
            if values currently do not exist for **context**

        COMMENT:
        REPLACE WITH EVC/OCM EXAMPLE
        Examples
        --------

        This figure shows an animation of the Composition in the XXX example script, with
        the `show_graph <ShowGraph.show_graph>` **show_learning** argument specified as *ALL*:

        .. _Composition_XXX_movie:

        .. figure:: _static/XXX_movie.gif
           :alt: Animation of Composition in XXX example script

        This figure shows an animation of the Composition in the XXX example script, with the `show_graph
        <ShowGraph.show_graph>` **show_control** argument specified as *ALL* and *UNIT* specified as *EXECUTION_SET*:

        .. _Composition_XXX_movie:

        .. figure:: _static/XXX_movie.gif
           :alt: Animation of Composition in XXX example script
        COMMENT

        Returns
        ---------

        2d list of values of OUTPUT Nodes at end of last trial : list[list]
          each item in the list is the `output_values <Mechanism_Base.output_values>` for an `OUTPUT` `Node
          <Composition_Nodes>` of the Composition, listed in the order listed in `get_nodes_by_role
          <Composition.get_nodes_by_role>`\(`NodeRole.OUTPUT <OUTPUT>`).

          .. note::
            The `results <Composition.results>` attribute of the Composition contains a list of the outputs for all
            trials.

        """
        # MODIFIED 3/28/22 OLD:
        context.source = ContextFlags.COMPOSITION
        # MODIFIED 3/28/22 END
        execution_phase = context.execution_phase
        context.execution_phase = ContextFlags.PREPARING

        for node in self.nodes:
            num_execs = node.parameters.num_executions._get(context)
            if num_execs is None:
                node.parameters.num_executions._set(Time(), context)
            else:
                node.parameters.num_executions._get(context)._set_by_time_scale(TimeScale.RUN, 0)

        if ContextFlags.SIMULATION_MODE not in context.runmode:
            try:
                self.parameters.input_specification._set(copy(inputs), context)
            except:
                self.parameters.input_specification._set(inputs, context)

        # May be used by controller for specifying num_trials_per_simulation
        self.num_trials = num_trials

        # Check to see if any Components are still in deferred init. If so, attempt to initialize them.
        # If they can not be initialized, raise a warning.
        self._complete_init_of_partially_initialized_nodes(context=context)

        if ContextFlags.SIMULATION_MODE not in context.runmode:
            self._check_controller_initialization_status()
            self._check_nodes_initialization_status()

            if not skip_analyze_graph:
                self._analyze_graph(context=context)

        if self._need_check_for_unused_projections:
            self._check_for_unused_projections(context=context)

        if scheduler is None:
            scheduler = self.scheduler

        if scheduling_mode is not None:
            scheduler.mode = scheduling_mode

        if default_absolute_time_unit is not None:
            scheduler.default_absolute_time_unit = default_absolute_time_unit

        self._check_for_unnecessary_feedback_projections()
        self._check_for_nesting_with_absolute_conditions(scheduler, termination_processing)

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
        results = self.parameters.results._get(context)
        if results is None:
            results = []

        self.rich_diverted_reports = None
        self.recorded_reports = None

        self._assign_execution_ids(context)

        scheduler._init_counts(execution_id=context.execution_id)

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

        inputs, num_inputs_sets = self._parse_run_inputs(inputs, context)

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
            and (context is None or ContextFlags.SIMULATION_MODE not in context.runmode)):
            self._initialize_from_context(context, base_context, override=False)

        context.composition = self

        if initialize_cycle_values is not None:
            self.initialize(values=initialize_cycle_values, include_unspecified_nodes=False, context=context)

        if not reset_stateful_functions_to:
            reset_stateful_functions_to = {}

        for node, vals in reset_stateful_functions_to.items():
            try:
                iter(vals)
            except TypeError:
                vals = [vals]
                reset_stateful_functions_to[node] = vals
            if (isinstance(reset_stateful_functions_when, Never) or
                    node not in reset_stateful_functions_when) and \
                    isinstance(node.reset_stateful_function_when, Never):
                try:
                    node.reset(**vals, context=context)
                except TypeError:
                    node.reset(*vals, context=context)

        # cache and set reset_stateful_function_when conditions for nodes, matching old System behavior
        # Validate
        valid_reset_type = True
        if not isinstance(reset_stateful_functions_when, (Condition, dict)):
            valid_reset_type = False
        elif type(reset_stateful_functions_when) == dict:
            if False in {True if isinstance(k, Mechanism) and isinstance(v, Condition) else
                         False for k,v in reset_stateful_functions_when.items()}:
                valid_reset_type = False

        if not valid_reset_type:
            raise CompositionError(
                f"{reset_stateful_functions_when} is not a valid specification for reset_integrator_nodes_when "
                f"of {self.name}. reset_integrator_nodes_when must be a Condition or a dict comprised of "
                f" {Node: Condition} pairs.")

        self._reset_stateful_functions_when_cache = {}

        # use type here to avoid another costly call to isinstance
        if not type(reset_stateful_functions_when) == dict:
            for node in self.nodes:
                try:
                    if isinstance(node.reset_stateful_function_when, Never):
                        self._reset_stateful_functions_when_cache[node] = node.reset_stateful_function_when
                        node.reset_stateful_function_when = reset_stateful_functions_when
                except AttributeError:
                    pass
        else:
            for node in reset_stateful_functions_when:
                self._reset_stateful_functions_when_cache[node] = node.reset_stateful_function_when
                node.reset_stateful_function_when = reset_stateful_functions_when[node]

        is_simulation = (context is not None and
                         ContextFlags.SIMULATION_MODE in context.runmode)

        if execution_mode & pnlvm.ExecutionMode._Run:
            # There's no mode to run simulations.
            # Simulations are run as part of the controller node wrapper.
            assert not is_simulation
            try:
                comp_ex_tags = frozenset({"learning"}) if self._is_learning(context) else frozenset()
                _comp_ex = pnlvm.CompExecution.get(self, context, additional_tags=comp_ex_tags)
                if execution_mode & pnlvm.ExecutionMode.LLVM:
                    results += _comp_ex.run(inputs, num_trials, num_inputs_sets)
                elif execution_mode & pnlvm.ExecutionMode.PTX:
                    results += _comp_ex.cuda_run(inputs, num_trials, num_inputs_sets)
                else:
                    assert False, "Unknown execution mode: {}".format(execution_mode)

                # Update the parameter for results
                self.parameters.results._set(results, context)

                if self._is_learning(context):
                    # copies back matrix to pnl from param struct (after learning)
                    _comp_ex._copy_params_to_pnl(context=context)

                self._propagate_most_recent_context(context)
                # KAM added the [-1] index after changing Composition run()
                # behavior to return only last trial of run (11/7/18)
                return results[-1]

            except Exception as e:
                if not execution_mode & pnlvm.ExecutionMode._Fallback:
                    raise e from None

                warnings.warn("Failed to run `{}': {}".format(self.name, str(e)))

        # Reset gym forager environment for the current trial
        if self.env:
            trial_output = np.atleast_2d(self.env.reset())
        else:
            trial_output = None

        context.execution_phase = execution_phase

        # EXECUTE TRIALS -------------------------------------------------------------

        with Report(self,
                    report_output=report_output,
                    report_params=report_params,
                    report_progress=report_progress,
                    report_simulations=report_simulations,
                    report_to_devices=report_to_devices,
                    context=context) as report:

            report_num = report.start_report(self, num_trials, context)

            report(self,
                   EXECUTE_REPORT,
                   report_num=report_num,
                   scheduler=scheduler,
                   content='run_start',
                   context=context)

            # Loop over the length of the list of inputs - each input represents a TRIAL
            for trial_num in range(num_trials):

                # Execute call before trial "hook" (user defined function)
                if call_before_trial:
                    call_with_pruned_args(call_before_trial, context=context)

                try:
                    run_term_cond = termination_processing[TimeScale.RUN]
                except (TypeError, KeyError):
                    run_term_cond = self.termination_processing[TimeScale.RUN]

                if run_term_cond.is_satisfied(
                    scheduler=scheduler,
                    context=context
                ):
                    break

                # PROCESSING ------------------------------------------------------------------------
                # Prepare stimuli from the outside world  -- collect the inputs for this TRIAL and store them in a dict
                try:
                    execution_stimuli = self._parse_trial_inputs(inputs, trial_num)
                except StopIteration:
                    break

                # execute processing, passing stimuli for this trial
                trial_output = self.execute(inputs=execution_stimuli,
                                            scheduler=scheduler,
                                            termination_processing=termination_processing,
                                            call_before_time_step=call_before_time_step,
                                            call_before_pass=call_before_pass,
                                            call_after_time_step=call_after_time_step,
                                            call_after_pass=call_after_pass,
                                            reset_stateful_functions_to=reset_stateful_functions_to,
                                            context=context,
                                            base_context=base_context,
                                            clamp_input=clamp_input,
                                            runtime_params=runtime_params,
                                            skip_initialization=True,
                                            execution_mode=execution_mode,
                                            report=report,
                                            report_num=report_num
                                            )

                # ---------------------------------------------------------------------------------
                # store the result of this execution in case it will be the final result

                # object.results.append(result)
                if isinstance(trial_output, collections.abc.Iterable):
                    result_copy = trial_output.copy()
                else:
                    result_copy = trial_output

                if ContextFlags.SIMULATION_MODE not in context.runmode:
                    results.append(result_copy)
                    self.parameters.results._set(results, context)

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

            # IMPLEMENTATION NOTE:
            # The AFTER Run controller execution takes place here, because there's no way to tell from within the
            # execute method whether or not we are at the last trial of the run.
            # The BEFORE Run controller execution takes place in the execute method,
            # because we can't execute the controller until after setup has occurred for the Input CIM.
            if (self.controller_mode == AFTER and
                self.controller_time_scale == TimeScale.RUN):
                try:
                    _comp_ex
                except NameError:
                    _comp_ex = None
                self._execute_controller(
                    execution_mode=execution_mode,
                    _comp_ex=_comp_ex,
                    report=report,
                    report_num=report_num,
                    context=context
                )

            report(self,
                   [RUN_REPORT, PROGRESS_REPORT],
                   report_num=report_num,
                   scheduler=scheduler,
                   content='run_end',
                   context=context,
                   node=self)

            # Reset input spec for next trial
            self.parameters.input_specification._set(None, context)

            scheduler.get_clock(context)._increment_time(TimeScale.RUN)

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

            # Undo override of reset_stateful_function_when conditions
            for node in self.nodes:
                try:
                    node.reset_stateful_function_when = self._reset_stateful_functions_when_cache[node]
                except KeyError:
                    pass

            return trial_output

    @handle_external_context()
    def learn(
            self,
            inputs: dict,
            targets: tc.optional(dict) = None,
            num_trials: tc.optional(int) = None,
            epochs: int = 1,
            minibatch_size: int = 1,
            patience: tc.optional(int) = None,
            min_delta: int = 0,
            context: tc.optional(Context) = None,
            execution_mode:pnlvm.ExecutionMode = pnlvm.ExecutionMode.Python,
            randomize_minibatches=False,
            call_before_minibatch = None,
            call_after_minibatch = None,
            *args,
            **kwargs
            ):
        """
            Runs the composition in learning mode - that is, any components with disable_learning False will be
            executed in learning mode. See `Composition_Learning` for details.

            Arguments
            ---------

            inputs: {`Node <Composition_Nodes>`:list }
                a dictionary containing a key-value pair for each `Node <Composition_Nodes>` (Mechanism or Composition)
                in the composition that receives inputs from the user. There are several equally valid ways that this
                dict can be structured:

                1. For each pair, the key is the  and the value is an input, the shape of which must match the Node's
                   default variable. This is identical to the input dict in the `run <Composition.run>` method
                   (see `Composition_Input_Dictionary` for additional details).

                2. A dict with keys 'inputs', 'targets', and 'epochs'. The `inputs` key stores a dict that is the same
                   same structure as input specification (1) of learn. The `targets` and `epochs` keys should contain
                   values of the same shape as `targets <Composition.learn>` and `epochs <Composition.learn>`.

            targets: {`Node <Composition_Nodes>`:list }
                a dictionary containing a key-value pair for each `Node <Composition_Nodes>` in the Composition that
                receives target values as input to the Composition for training `learning pathways
                <Composition_Learning_Pathway>`. The key of each entry can be either the `TARGET_MECHANISM
                <Composition_Learning_Components>` for a learning pathway or the final Node in that Pathway, and
                the value is the target value used for that Node on each trial (see `target inputs
                <Composition_Target_Inputs>` for additional details concerning the formatting of targets).

            num_trials : int (default=None)
                typically, the Composition infers the number of trials to execute from the length of its input
                specification.  However, **num_trials** can be used to enforce an exact number of trials to execute;
                if it is greater than there are inputs then inputs will be repeated (see `Composition_Execution_Inputs`
                for additional information).

            epochs : int (default=1)
                specifies the number of training epochs (that is, repetitions of the batched input set) to run with

            minibatch_size : int (default=1)
                specifies the size of the minibatches to use. The input trials will be batched and ran, after which
                learning mechanisms with learning mode TRIAL will update weights

            randomize_minibatch: bool (default=False)
                specifies whether the order of the input trials should be randomized on each epoch

            patience : int or None (default=None)
                used for early stopping of training; If a model has more than `patience` bad consecutive epochs,
                then `learn` will prematurely return. A bad epoch is determined by the `min_delta` value

            min_delta : float (default=0)
                the minimum reduction in average loss that an epoch must provide in order to qualify as a 'good' epoch;
                Any reduction less than this value is considered to be a bad epoch.
                Used for early stopping of training, in combination with `patience`.

            scheduler : Scheduler
                the scheduler object that owns the conditions that will instruct the execution of the Composition
                If not specified, the Composition will use its automatically generated scheduler.

            context
                context will be set to self.default_execution_id if unspecified

            call_before_minibatch : callable
                called before each minibatch is executed

            call_after_minibatch : callable
                called after each minibatch is executed

            report_output : ReportOutput : default ReportOutput.OFF
                specifies whether to show output of the Composition and its `Nodes <Composition_Nodes>` trial-by-trial
                as it is generated; see `Report_Output` for additional details and `ReportOutput` for options.

            report_params : ReportParams : default ReportParams.OFF
                specifies whether to show values the `Parameters` of the Composition and its `Nodes <Composition_Nodes>`
                as part of the output report; see `Report_Output` for additional details and `ReportParams` for options.

            report_progress : ReportProgress : default ReportProgress.OFF
                specifies whether to report progress of execution in real time; see `Report_Progress` for additional
                details.

            report_simulations : ReportSimulatons : default ReportSimulations.OFF
                specifies whether to show output and/or progress for `simulations
                <OptimizationControlMechanism_Execution>` executed by the Composition's `controller
                <Composition_Controller>`; see `Report_Simulations` for additional details.

            report_to_devices : list(ReportDevices) : default ReportDevices.CONSOLE
                specifies where output and progress should be reported; see `Report_To_Devices` for additional
                details and `ReportDevices` for options.

            Returns
            ---------

            the results of the final epoch of training : list
        """
        from psyneulink.library.compositions import CompositionRunner
        runner = CompositionRunner(self)

        context.add_flag(ContextFlags.LEARNING_MODE)
        # # MODIFIED 3/28/22 NEW:
        # context.source = ContextFlags.COMPOSITION
        # MODIFIED 3/28/22 END
        # # FIX 5/28/20
        # context.add_flag(ContextFlags.PREPARING)
        # context.execution_phase=ContextFlags.PREPARING

        self._analyze_graph()

        # Temporary check to ensure that nested compositions don't have stranded target Mechanisms.
        # This should be taken out once we automatically instantiate Mechs to project to nested target Mechs.
        nc = self._get_nested_compositions()
        for comp in nc:
            nc_targets = [path.target for path in comp.pathways if path.target]
            for target in nc_targets:
                target_mech_input_cim_input_port = comp.input_CIM.port_map.get(target.input_port)[0]
                if not target_mech_input_cim_input_port.path_afferents:
                    raise CompositionError(
                        f'Target mechanism {target.name} of nested Composition {comp.name} is not being projected to '
                        f'from its enclosing Composition {self.name}. For a call to {self.name}.learn, {target.name} '
                        f'must have an afferent projection with a target value so that an error term may be computed. '
                        f'A reference to {target.name}, with which you can create the needed projection, can be found '
                        f'as the target attribute of the relevant pathway in {comp.name}.pathways. '
                    )

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
            execution_mode=execution_mode,
            *args, **kwargs)

        context.remove_flag(ContextFlags.LEARNING_MODE)
        return learning_results

    def _execute_controller(self,
                            relative_order=AFTER,
                            execution_mode=False,
                            _comp_ex=False,
                            report=None,
                            report_num=None,
                            context=None
                            ):
        execution_scheduler = context.composition.scheduler
        if (self.enable_controller and
            self.controller_mode == relative_order and
            self.controller_condition.is_satisfied(scheduler=execution_scheduler,
                                                   context=context)
        ):

            # control phase
            # FIX: SHOULD SET CONTEXT AS CONTROL HERE AND RESET AT END (AS DONE FOR animation BELOW)
            if (
                    self.initialization_status != ContextFlags.INITIALIZING
                    and ContextFlags.SIMULATION_MODE not in context.runmode
            ):

                # Report controller engagement before executing simulations
                #    so it appears before them for ReportOutput.TERSE
                report(self,
                       EXECUTE_REPORT,
                       report_num=report_num,
                       scheduler=execution_scheduler,
                       content='controller_start',
                       context=context,
                       node=self.controller)

                if self.controller and not execution_mode:
                    context.execution_phase = ContextFlags.PROCESSING
                    self.controller.execute(context=context)

                if execution_mode:
                    _comp_ex.execute_node(self.controller, context=context)

                context.remove_flag(ContextFlags.PROCESSING)

                # Animate controller (before execution)
                context.execution_phase = ContextFlags.CONTROL
                if self._animate is not False and SHOW_CONTROLLER in self._animate and self._animate[SHOW_CONTROLLER]:
                    self._animate_execution(self.controller, context)
                context.remove_flag(ContextFlags.CONTROL)

                # Report controller execution after executing simulations
                #    so it includes the results for ReportOutput.FULL
                report(self,
                       CONTROLLER_REPORT,
                       report_num=report_num,
                       scheduler=execution_scheduler,
                       content='controller_end',
                       context=context,
                       node=self.controller)

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
            reset_stateful_functions_to=None,
            context=None,
            base_context=Context(execution_id=None),
            clamp_input=SOFT_CLAMP,
            runtime_params=None,
            skip_initialization=False,
            execution_mode:pnlvm.ExecutionMode = pnlvm.ExecutionMode.Python,
            report_output:ReportOutput=ReportOutput.OFF,
            report_params:ReportOutput=ReportParams.OFF,
            report_progress:ReportProgress=ReportProgress.OFF,
            report_simulations:ReportSimulations=ReportSimulations.OFF,
            report_to_devices:ReportDevices=None,
            report=None,
            report_num=None,
            ):
        """
            Passes inputs to any `Nodes <Composition_Nodes>` receiving inputs directly from the user (via the "inputs"
            argument) then coordinates with the `Scheduler` to execute sets of Nodes that are eligible to execute until
            `termination conditions <Scheduler_Termination_Conditions>` are met.

            Arguments
            ---------

            inputs: { `Node <Composition_Nodes>`: list } : default None
                a dictionary containing a key-value pair for each `Node <Composition_Nodes>` in the Composition that
                receives inputs from the user. For each pair, the key is the `Node <Composition_Nodes>` (a `Mechanism
                <Mechanism>` or `Composition`) and the value is an input, the shape of which must match the Node's
                default variable. If **inputs** is not specified, the `default_variable <Component_Variable>`
                for each `INPUT` Node is used as its input (see `Input Formats <Composition_Execution_Inputs>` for
                additional details).

            clamp_input : SOFT_CLAMP : default SOFT_CLAMP

            runtime_params : Dict[Node: Dict[Parameter: Tuple(Value, Condition)]] : default None
                specifies alternate parameter values to be used only during this `EXECUTION` when the specified
                `Condition` is met (see `Composition_Runtime_Params` for more details and examples of valid
                dictionaries).

            skip_initialization :  : default False
                COMMENT:
                    NEEDS DESCRIPTION
                COMMENT

            scheduler : Scheduler : default None
                the scheduler object that owns the conditions that will instruct the execution of the Composition
                If not specified, the Composition will use its automatically generated scheduler.

            context : `execution_id <Context.execution_id>` : default `default_execution_id`
                `execution context <Composition_Execution_Context>` in which the `Composition` will be executed.

            base_context : `execution_id <Context.execution_id>` : Context(execution_id=None)
                the context corresponding to the `execution context <Composition_Execution_Context>` from which this
                execution will be initialized, if values currently do not exist for **context**.

            call_before_time_step : callable : default None
                called before each `TIME_STEP` is executed
                passed the current *context* (but it is not necessary for your callable to take).

            call_after_time_step : callable : default None
                called after each `TIME_STEP` is executed
                passed the current *context* (but it is not necessary for your callable to take).

            call_before_pass : callable : default None
                called before each `PASS` is executed
                passed the current *context* (but it is not necessary for your callable to take).

            call_after_pass : callable : default None
                called after each `PASS` is executed
                passed the current *context* (but it is not necessary for your callable to take).

            execution_mode : enum.Enum[Auto|LLVM|LLVMexec|Python|PTXExec] : default Python
                specifies whether to run using the Python interpreter or a `compiled mode <Composition_Compilation>`.
                see **execution_mode** argument of `run <Composition.run>` method for additional details.

            report_output : ReportOutput : default ReportOutput.OFF
                specifies whether to show output of the Composition and its `Nodes <Composition_Nodes>` for the
                execution; see `Report_Output` for additional details and `ReportOutput` for options.

            report_params : ReportParams : default ReportParams.OFF
                specifies whether to show values the `Parameters` of the Composition and its `Nodes <Composition_Nodes>`
                for the execution; see `Report_Output` for additional details and `ReportParams` for options.

            report_progress : ReportProgress : default ReportProgress.OFF
                specifies whether to report progress of the execution; see `Report_Progress` for additional details.

            report_simulations : ReportSimulations : default ReportSimulations.OFF
                specifies whether to show output and/or progress for `simulations
                <OptimizationControlMechanism_Execution>` executed by the Composition's `controller
                <Composition_Controller>`; see `Report_Simulations` for additional details.

            report_to_devices : list(ReportDevices) : default ReportDevices.CONSOLE
                specifies where output and progress should be reported; see `Report_To_Devices` for additional
                details and `ReportDevices` for options.

            Returns
            ---------
            output_values : List
            These are the values of the Composition's output_CIM.output_ports, excluding those the source of which
            are from a (potentially nested) Node with NodeRole.PROBE in its enclosing Composition.
        """

        with Report(self,
                    report_output=report_output,
                    report_params=report_params,
                    report_progress=report_progress,
                    report_simulations=report_simulations,
                    report_to_devices=report_to_devices,
                    context=context) as report:

            execution_scheduler = scheduler or self.scheduler

            # TODO: scheduler counts and clocks were not expected to be
            # used prior to Scheduler.run calls. Remove this hack when
            # accommodation is written
            try:
                execution_scheduler._init_counts(context.execution_id, base_context.execution_id)
            except graph_scheduler.SchedulerError:
                execution_scheduler._init_counts(context.execution_id)

            # If execute method is called directly, need to create Report object for reporting
            if not (context.source & ContextFlags.COMPOSITION) or report_num is None:
                report_num = report.start_report(comp=self, num_trials=1, context=context)

                # Also, call report to generate initial line of report
                if context.source is ContextFlags.COMMAND_LINE:
                    report(self,
                           EXECUTE_REPORT,
                           report_num=report_num,
                           scheduler=execution_scheduler,
                           content='execute_start',
                           context=context
                           )

            # ASSIGNMENTS **********************************************************************************************

            if not hasattr(self, '_animate'):
                # These are meant to be assigned in run method;  needed here for direct call to execute method
                self._animate = False

            # IMPLEMENTATION NOTE:
            # KAM 4/29/19
            # The nested var is set to True if the Composition is nested in another Composition, otherwise False
            # Later on, this is used to determine:
            #   (1) whether to initialize from context
            #   (2) whether to assign values to CIM from input dict (if not nested) or simply execute CIM (if nested)
            # JDC 3/28/22:
            #    This currently prevents a Composition that is nested within another to be tested on its own
            #    Would be good to figure out a way to accomodate that
            nested = False
            if len(self.input_CIM.path_afferents) > 0:
                nested = True

            runtime_params = self._parse_runtime_params_conditions(runtime_params)

            # Assign the same execution_ids to all nodes in the Composition and get it (if it was None)
            self._assign_execution_ids(context)

            context.composition = self

            input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

            # if execute was called from command line and no inputs were specified,
            # assign default inputs to highest level composition (i.e. not on any nested Compositions)
            if not inputs and not nested and ContextFlags.COMMAND_LINE in context.source:
                inputs = self._instantiate_input_dict({})
            # Skip initialization if possible (for efficiency):
            # - and(context has not changed
            # -     structure of the graph has not changed
            # -     not a nested composition
            # -     its not a simulation)
            # - or(gym forage env is being used)
            # (e.g., when run is called externally repeated for the same environment)
            # KAM added HACK below "or self.env is None" to merge in interactive inputs fix for speed improvement
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
                    and context.execution_phase is not ContextFlags.SIMULATION_MODE
                ):
                    self._initialize_from_context(context, base_context, override=False)
                    context.composition = self

            # Run compiled execution (if compiled execution was requested
            # NOTE: This should be as high up as possible,
            # but still after the context has been initialized
            if execution_mode:
                is_simulation = (context is not None and
                                 ContextFlags.SIMULATION_MODE in context.runmode)
                # Try running in Exec mode first
                if (execution_mode & pnlvm.ExecutionMode._Exec):
                    # There's no mode to execute simulations.
                    # Simulations are run as part of the controller node wrapper.
                    assert not is_simulation
                    try:
                        llvm_inputs = self._validate_execution_inputs(inputs)
                        _comp_ex = pnlvm.CompExecution.get(self, context)
                        if execution_mode & pnlvm.ExecutionMode.LLVM:
                            _comp_ex.execute(llvm_inputs)
                        elif execution_mode & pnlvm.ExecutionMode.PTX:
                            _comp_ex.cuda_execute(llvm_inputs)
                        else:
                            assert False, "Unknown execution mode: {}".format(execution_mode)

                        report(self,
                               PROGRESS_REPORT,
                               report_num=report_num,
                               content='trial_end',
                               context=context)

                        self._propagate_most_recent_context(context)
                        return _comp_ex.extract_node_output(self.output_CIM)

                    except Exception as e:
                        if not execution_mode & pnlvm.ExecutionMode._Fallback:
                            raise e from None

                        warnings.warn("Failed to execute `{}': {}".format(self.name, str(e)))

                # Exec failed for some reason, we can still try node level execution_mode
                # Filter out nested compositions. They are not executed in this mode
                # Filter out controller if running simulation.
                mechanisms = (n for n in self._all_nodes
                              if isinstance(n, Mechanism) and
                                 (n is not self.controller or not is_simulation))

                assert execution_mode & pnlvm.ExecutionMode.LLVM
                try:
                    _comp_ex = pnlvm.CompExecution.get(self, context)
                    # Compile all mechanism wrappers
                    for m in mechanisms:
                        _comp_ex._set_bin_node(m)
                except Exception as e:
                    if not execution_mode & pnlvm.ExecutionMode._Fallback:
                        raise e from None

                    warnings.warn("Failed to compile wrapper for `{}' in `{}': {}".format(m.name, self.name, str(e)))
                    execution_mode = pnlvm.ExecutionMode.Python


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

            # Set num_executions.TRIAL to 0 for every node
            # Reset any nodes that have satisfied 'reset_stateful_function_when' conditions.
            if not reset_stateful_functions_to:
                reset_stateful_functions_to = {}

            for node in self.nodes:
                node.parameters.num_executions.get(context)._set_by_time_scale(TimeScale.TRIAL, 0)
                if node.parameters.has_initializers._get(context):
                    try:
                        if (
                            node.reset_stateful_function_when.is_satisfied(
                                scheduler=execution_scheduler,
                                context=context
                            )
                        ):
                            vals = reset_stateful_functions_to.get(node, [None])
                            try:
                                node.reset(**vals, context=context)
                            except TypeError:
                                node.reset(*vals, context=context)
                    except AttributeError:
                        pass

            # FIX 5/28/20
            context.remove_flag(ContextFlags.PREPARING)

            # EXECUTE INPUT CIM ****************************************************************************************

            # FIX: 6/12/19 MOVE TO EXECUTE BELOW?
            # Handles Input CIM and Parameter CIM execution.
            #
            # FIX: 8/21/19
            # If self is a nested composition, its input CIM will obtain its value in one of two ways,
            # depending on whether or not it is being executed within a simulation.
            # If it is a simulation, then we need to use the _build_variable_for_input_CIM method, which parses the
            # inputs argument of the execute method into a suitable shape for the input ports of the input_CIM.
            # If it is not a simulation, we can simply execute the input CIM.
            #
            # If self is an unnested composition, we must update the input ports for any input nodes that are
            # Compositions. This is done to update the variable for their input CIMs, which allows the
            # _adjust_execution_stimuli method to properly validate input for those nodes.
            # -DS

            context.execution_phase = ContextFlags.PROCESSING
            if inputs is not None:
                inputs = self._validate_execution_inputs(inputs)
                build_CIM_input = self._build_variable_for_input_CIM(inputs)

            if execution_mode:
                _comp_ex.execute_node(self.input_CIM, inputs, context)
                # FIXME: parameter_CIM should be executed here as well,
                #        but node execution of nested compositions with
                #        outside control is not supported yet.
                assert not nested or len(self.parameter_CIM.afferents) == 0

            elif nested:

                # MODIFIED 3/28/22 CURRENT:
                # IMPLEMENTATION NOTE: context.string set in Mechanism.execute
                direct_call = (f"{context.source.name} EXECUTING" not in context.string)
                # MODIFIED 3/28/22 NEW:
                # direct_call = (context.source == ContextFlags.COMMAND_LINE)
                # MODIFIED 3/28/22 END
                simulation = ContextFlags.SIMULATION_MODE in context.runmode
                if simulation or direct_call:
                    # For simulations, or direct call to nested Composition (e.g., from COMMAND_LINE to test it)
                    #  assign inputs if they not provided (e.g., # autodiff)
                    if inputs is not None:
                        self.input_CIM.execute(build_CIM_input, context=context)
                    else:
                        self.input_CIM.execute(context=context)
                else:
                    # regular run (DEFAULT_MODE) of nested Composition called from enclosing Composition,
                    #    so inputs should be None, and be assigned from nested Composition's input_CIM
                    assert inputs is None,\
                        f"Input provided to a nested Composition {self.name} in call from outer composition."
                    self.input_CIM.execute(context=context)

                self.parameter_CIM.execute(context=context)

            else:
                self.input_CIM.execute(build_CIM_input, context=context)

                # Update nested compositions
                for comp in (node for node in self.get_nodes_by_role(NodeRole.INPUT) if isinstance(node, Composition)):
                    for port in comp.input_ports:
                        port._update(context=context)

            # FIX: 6/12/19 Deprecate?
            # Manage input clamping

            # 1 because call_before_pass is called before the main
            # scheduler loop to ensure it happens regardless of whether
            # the scheduler terminates a trial immediately
            next_pass_before = 1
            next_pass_after = 1
            last_pass = None

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

            # EXECUTE CONTROLLER (if specified for BEFORE) *************************************************************

            # Execute controller --------------------------------------------------------

            try:
                _comp_ex
            except NameError:
                _comp_ex = None
            # IMPLEMENTATION NOTE:
            # The BEFORE Run controller execution takes place here, because we can't execute the controller until after
            # setup has occurred for the Input CIM, whereas the AFTER Run controller execution takes place in the run
            # method, because there's no way to tell from within the execute method whether or not we are at the last
            # trial of the run.
            if self.controller_time_scale == TimeScale.RUN and scheduler.get_clock(context).time.trial == 0:
                self._execute_controller(
                    relative_order=BEFORE,
                    execution_mode=execution_mode,
                    _comp_ex=_comp_ex,
                    report=report,
                    report_num=report_num,
                    context=context
                )
            elif self.controller_time_scale == TimeScale.TRIAL:
                self._execute_controller(
                    relative_order=BEFORE,
                    execution_mode=execution_mode,
                    _comp_ex=_comp_ex,
                    report=report,
                    report_num=report_num,
                    context=context
                )

            # EXECUTE EACH EXECUTION SET *******************************************************************************

            # Begin reporting of TRIAL:
            # - add TRIAL header and Composition's input to output report (now that they are known)
            report(self,
                   EXECUTE_REPORT,
                   report_num=report_num,
                   scheduler=execution_scheduler,
                   content='trial_start',
                   context=context
                   )

            # PREPROCESS (get inputs, call_before_pass, animate first frame) ----------------------------------

            context.execution_phase = ContextFlags.PROCESSING

            try:
                _comp_ex
            except NameError:
                _comp_ex = None

            if call_before_pass:
                call_with_pruned_args(call_before_pass, context=context)

            if self.controller_time_scale == TimeScale.PASS:
                self._execute_controller(
                    relative_order=BEFORE,
                    execution_mode=execution_mode,
                    _comp_ex=_comp_ex,
                    report=report,
                    report_num=report_num,
                    context=context
                )

            # GET execution_set -------------------------------------------------------------------------
            # run scheduler to receive sets of nodes that may be executed at this time step in any order
            execution_sets = execution_scheduler.run(termination_conds=termination_processing,
                                                          context=context,
                                                          skip_trial_time_increment=True,
                                                          )
            if context.runmode == ContextFlags.SIMULATION_MODE:
                for i in range(scheduler.get_clock(context).time.time_step):
                    execution_sets.__next__()

            for next_execution_set in execution_sets:

                # SETUP EXECUTION ----------------------------------------------------------------------------

                # IMPLEMENTATION NOTE KDM 1/15/20:
                # call_*after*_pass is called here because we can't tell at the end of this code block whether a PASS
                # has ended or not. The scheduler only modifies the pass after we receive an execution_set. So, we only
                # know a PASS has ended in retrospect after the scheduler has changed the clock to indicate it. So, we
                # have to run call_after_pass before the next PASS (here) or after this code block (see call to
                # call_after_pass below)
                curr_pass = execution_scheduler.get_clock(context).get_total_times_relative(TimeScale.PASS,
                                                                                            TimeScale.TRIAL)
                new_pass = False
                if curr_pass != last_pass:
                    new_pass = True
                    last_pass = curr_pass
                if next_pass_after == curr_pass:
                    if call_after_pass:
                        logger.debug(f'next_pass_after {next_pass_after}\tscheduler pass {curr_pass}')
                        call_with_pruned_args(call_after_pass, context=context)
                    if self.controller_time_scale == TimeScale.PASS:
                        self._execute_controller(
                            relative_order=AFTER,
                            execution_mode=execution_mode,
                            _comp_ex=_comp_ex,
                            report=report,
                            report_num=report_num,
                            context=context
                        )
                    next_pass_after += 1
                if next_pass_before == curr_pass:
                    if call_before_pass:
                        call_with_pruned_args(call_before_pass, context=context)
                        logger.debug(f'next_pass_before {next_pass_before}\tscheduler pass {curr_pass}')
                    if self.controller_time_scale == TimeScale.PASS:
                        self._execute_controller(
                            relative_order=BEFORE,
                            execution_mode=execution_mode,
                            _comp_ex=_comp_ex,
                            report=report,
                            report_num=report_num,
                            context=context
                        )
                    next_pass_before += 1

                if call_before_time_step:
                    call_with_pruned_args(call_before_time_step, context=context)

                if self.controller_time_scale == TimeScale.TIME_STEP:
                    self._execute_controller(
                        relative_order=BEFORE,
                        execution_mode=execution_mode,
                        _comp_ex=_comp_ex,
                        report=report,
                        report_num=report_num,
                        context=context
                    )

                # MANAGE EXECUTION OF FEEDBACK / CYCLIC GRAPHS ------------------------------------------------
                # Set up storage of all node values *before* the start of each timestep
                # If nodes within a timestep are connected by projections, those projections must pass their senders'
                # values from the beginning of the timestep (i.e. their "frozen values")
                # This ensures that the order in which nodes execute does not affect the results of this timestep
                frozen_values = {}
                new_values = {}
                if execution_mode:
                    _comp_ex.freeze_values()

                # PURGE LEARNING IF NOT ENABLED ----------------------------------------------------------------
                # If learning is turned off, check for learning related nodes and remove them from the execution set
                if not self._is_learning(context):
                    next_execution_set = next_execution_set - set(self.get_nodes_by_role(NodeRole.LEARNING))

                # Add TIME_STEP header to output report
                nodes_to_report = any(node.reportOutputPref for node in next_execution_set)
                report(self,
                       EXECUTE_REPORT,
                       report_num=report_num,
                       scheduler=execution_scheduler,
                       content='time_step_start',
                       context=context,
                       nodes_to_report=True)

                # ANIMATE execution_set ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if self._animate is not False and self._animate_unit == EXECUTION_SET:
                    context.execution_phase = ContextFlags.PROCESSING
                    self._animate_execution(next_execution_set, context)

                # EXECUTE EACH NODE IN EXECUTION SET -------------------------------------------------------------------
                if execution_scheduler.mode is SchedulingMode.EXACT_TIME:
                    # sort flattened execution set by unflattened position
                    next_execution_set = sorted(
                        next_execution_set,
                        key=lambda n: execution_scheduler.consideration_queue_indices[n]
                    )

                # execute each node with EXECUTING in context
                for (node_idx, node) in enumerate(next_execution_set):

                    node.parameters.num_executions.get(context)._set_by_time_scale(TimeScale.TIME_STEP, 0)
                    if new_pass:
                        node.parameters.num_executions.get(context)._set_by_time_scale(TimeScale.PASS, 0)


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
                            execution_runtime_params.update(
                                self._get_satisfied_runtime_param_values(runtime_params[node],
                                                                         execution_scheduler,
                                                                         context))

                        # (Re)set context.execution_phase to PROCESSING by default
                        context.execution_phase = ContextFlags.PROCESSING

                        # Set to LEARNING if Mechanism receives any PathwayProjections that are being learned
                        #   for which learning_enabled == True or ONLINE (i.e., not False or AFTER)
                        #   Implementation Note: RecurrentTransferMechanisms are special cased as the
                        #   AutoAssociativeMechanism should be handling learning - not the RTM itself.
                        if self._is_learning(context) and not isinstance(node, RecurrentTransferMechanism):
                            projections = set(self.projections).intersection(set(node.path_afferents))
                            if any([p for p in projections if
                                    any([a for a in p.parameter_ports[MATRIX].mod_afferents
                                         if (hasattr(a, 'learning_enabled')
                                             and a.learning_enabled in {True, ONLINE})])]):
                                context.replace_flag(ContextFlags.PROCESSING, ContextFlags.LEARNING)

                        # Execute Mechanism
                        if execution_mode:
                            _comp_ex.execute_node(node, context=context)
                        else:
                            if node is not self.controller:
                                mech_context = copy(context)
                                mech_context.source = ContextFlags.COMPOSITION
                                if nested and node in self.get_nodes_by_role(NodeRole.INPUT):
                                    for port in node.input_ports:
                                        port._update(context=context)
                                node.execute(context=mech_context,
                                             report_num=report_num,
                                             runtime_params=execution_runtime_params,
                                             )

                        # Set execution_phase for node's context back to IDLE
                        if self._is_learning(context):
                            context.replace_flag(ContextFlags.LEARNING, ContextFlags.PROCESSING)
                        context.remove_flag(ContextFlags.PROCESSING)

                    # EXECUTE A NESTED COMPOSITION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    elif isinstance(node, Composition):

                        if execution_mode:
                            # Invoking nested composition passes data via Python
                            # structures. Make sure all sources get their latest values
                            srcs = (proj.sender.owner for proj in node.input_CIM.afferents)
                            for srnode in srcs:
                                if srnode is self.input_CIM or srnode in self.nodes:
                                    data_loc = srnode
                                else:
                                    # Consuming output from another nested composition
                                    assert srnode.composition in self.nodes
                                    assert srnode is srnode.composition.output_CIM
                                    data_loc = srnode.composition

                                # Set current Python values to LLVM results
                                data = _comp_ex.extract_frozen_node_output(data_loc)
                                for op, v in zip(srnode.output_ports, data):
                                    op.parameters.value._set(
                                      v, context, skip_history=True, skip_log=True)

                            # Update afferent projections and input ports.
                            node.input_CIM._update_input_ports(context=context)

                        # Pass outer context to nested Composition
                        context.composition = node
                        if ContextFlags.SIMULATION_MODE in context.runmode:
                            is_simulating = True
                            context.remove_flag(ContextFlags.SIMULATION_MODE)
                        else:
                            is_simulating = False

                        # Run node-level compiled nested composition
                        # only if there are no control projections
                        nested_execution_mode = execution_mode \
                            if len(node.parameter_CIM.afferents) == 0 else \
                            pnlvm.ExecutionMode.Python
                        ret = node.execute(context=context,
                                           execution_mode=nested_execution_mode)

                        # Get output info from nested execution
                        if execution_mode:
                            # Update result in binary data structure
                            _comp_ex.insert_node_output(node, ret)

                        if is_simulating:
                            context.add_flag(ContextFlags.SIMULATION_MODE)

                        context.composition = self

                        # Add Node info for TIME_STEP to output report
                        report(self,
                               EXECUTE_REPORT,
                               report_num=report_num,
                               scheduler=execution_scheduler,
                               content='nested_comp',
                               context=context,
                               node=node)

                    # ANIMATE node ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if self._animate is not False and self._animate_unit == COMPONENT:
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

                # Complete TIME_STEP entry for output report
                report(self,
                       EXECUTE_REPORT,
                       report_num=report_num,
                       scheduler=execution_scheduler,
                       content='time_step_end',
                       context=context,
                       nodes_to_report=nodes_to_report)

                if self.controller_time_scale == TimeScale.TIME_STEP:
                    self._execute_controller(
                        relative_order=AFTER,
                        execution_mode=execution_mode,
                        _comp_ex=_comp_ex,
                        report=report,
                        report_num=report_num,
                        context=context
                    )

                if call_after_time_step:
                    call_with_pruned_args(call_after_time_step, context=context)

            context.remove_flag(ContextFlags.PROCESSING)

            #Update matrix parameter of PathwayProjections being learned with learning_enabled==AFTER
            from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
            if self._is_learning(context) and not isinstance(self, AutodiffComposition):
                context.execution_phase = ContextFlags.LEARNING
                for projection in [p for p in self.projections if
                                   hasattr(p, 'has_learning_projection') and p.has_learning_projection]:
                    matrix_parameter_port = projection.parameter_ports[MATRIX]
                    if any([lp for lp in matrix_parameter_port.mod_afferents if lp.learning_enabled == AFTER]):
                        matrix_parameter_port._update(context=context)
                context.remove_flag(ContextFlags.LEARNING)

            if call_after_pass:
                call_with_pruned_args(call_after_pass, context=context)

            if self.controller_time_scale == TimeScale.PASS:
                self._execute_controller(
                    relative_order=AFTER,
                    execution_mode=execution_mode,
                    _comp_ex=_comp_ex,
                    report=report,
                    report_num=report_num,
                    context=context
                )

            # Reset context flags
            context.execution_phase = ContextFlags.PROCESSING
            self.output_CIM.execute(context=context)
            context.execution_phase = ContextFlags.IDLE

            # Animate output_CIM
            # FIX: NOT SURE WHETHER IT CAN BE LEFT IN PROCESSING AFTER THIS -
            #      COORDINATE WITH REFACTORING OF PROCESSING/CONTROL CONTEXT
            if self._animate is not False and SHOW_CIM in self._animate and self._animate[SHOW_CIM]:
                self._animate_execution(self.output_CIM, context)
            # FIX: END

            # Complete TRIAL Panel for output report, and report progress
            #  note: do so before executing controller, so that it appears after trial report if controller_mode=AFTER
            report(self,
                   [EXECUTE_REPORT, PROGRESS_REPORT],
                   report_num=report_num,
                   scheduler=execution_scheduler,
                   content='trial_end',
                   context=context)
            # # FIX: 3/28/21 ??MOVE TO VERY END?
            # report(self,
            #        PROGRESS_REPORT,
            #        report_num=report_num,
            #        context=context)

            # EXECUTE CONTROLLER (if controller_mode == AFTER) *********************************************************
            if self.controller_time_scale == TimeScale.TRIAL:
                self._execute_controller(
                    relative_order=AFTER,
                    execution_mode=execution_mode,
                    _comp_ex=_comp_ex,
                    report=report,
                    report_num=report_num,
                    context=context
                )

            # # If called directly, wrap up reporting
            if context.source is ContextFlags.COMMAND_LINE:
                report(self,
                       EXECUTE_REPORT,
                       report_num=report_num,
                       scheduler=execution_scheduler,
                       content='execute_end',
                       context=context)

            # Extract result here
            if execution_mode:
                _comp_ex.freeze_values()
                _comp_ex.execute_node(self.output_CIM, context=context)
                report(self,
                       PROGRESS_REPORT,
                       report_num=report_num,
                       content='trial_end',
                       context=context)
                return _comp_ex.extract_node_output(self.output_CIM)

            # UPDATE TIME and RETURN ***********************************************************************************

            execution_scheduler.get_clock(context)._increment_time(TimeScale.TRIAL)

            return self.get_output_values(context)

    def __call__(self, *args, **kwargs):
        """Execute Composition of any args are provided;  else simply return results of last execution.
        This allows Composition, after it has been constructed, to be run simply by calling it directly.
        """
        if not args and not kwargs:
            if self.results:
                return self.results[-1]
            else:
                return None
        elif (args and isinstance(args[0],dict)) or INPUTS in kwargs:
            from psyneulink.core.compositions.pathway import PathwayRole
            if any(PathwayRole.LEARNING in p.roles and p.target in kwargs[INPUTS] for p in self.pathways):
                return self.learn(*args, **kwargs)
            else:
                return self.run(*args, **kwargs)
        else:
            bad_args_str = ", ".join([str(arg) for arg in args] + list(kwargs.keys()))
            raise CompositionError(f"Composition ({self.name}) called with illegal argument(s): {bad_args_str}")

    def get_inputs_format(self, **kwargs):
        """Alias of get_input_format (easy mistake to make)"""
        return self.get_input_format(**kwargs, alias="get_inputs_format")

    def get_input_format(self,
                         form:Union[DICT,TEXT]=DICT,
                         num_trials:Union[int, FULL]=1,
                         use_labels:bool=False,
                         use_names:bool=False,
                         show_nested_input_nodes:bool=False,
                         alias:str=None):
        """Return dict or string with format of dict used by **inputs** argument of `run <Composition.run>` method.

        Arguments
        ---------

        form : DICT or TEXT : default DICT
            specifies the form in which the exampple is returned; DICT (the default) returns a dict (with
            **num_trials** worth of default values for each `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>`)
            formatted for use as the **inputs** arg of the Compositon's `run <Composition.run>` method;
            TEXT returns a user-readable text description of the format (optionally with inputs required for
            `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of any `nested Compositions <Composition_Nested>`
            (see **show_nested_input_nodes** below).

        num_trials : int or FULL : default 1
            specifies number of trials' worth of inputs to include in returned item.  If *FULL* is specified,
            **use_labels** is automatically set to True, and **num_trials** is set to number of labels in the
            `input_label_dict <Mechanism_Base.input_labels_dict>` with the largest number of labels specified; if
            none of the `INPUT <NodeRole.INPUT>` Mechanisms in the Composition (including any nested ones) have an
            `input_label_dict <Mechanism_Base.input_labels_dict>` specified, **num_trials** is set to the default (1).

        use_labels : bool : default False
            if True, shows labels instead of values for Mechanisms that have an `input_label_dict
            <Mechanism_Base.input_labels_dict>`.  For **num_trials** = 1, a representative label is
            shown; for **num_trials** > 1, a different label is used for each trial shown, cycling
            through the set if **num_trials** is greater than the number of labels.  If **num_trials = *FULL*,
            trials will be included.

            it is set to the number of labels in the largest list specified in any `input_label_dict
            <Mechanism_Base.input_labels_dict>` specified for an `INPUT <NodeRole.INPUT>` Mechanism;

        use_names : bool : default False
            use `Node <Composition_Nodes>` name as key for Node if **form** = DICT.

        show_nested_input_nodes : bool : default False
            show hierarchical display of `Nodes <Composition_Nodes>` in `nested Compositions <Composition_Nested>`
            with names of destination `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` and representative inputs,
            followed by the actual format used for the `run <Composition.run>` method.

        Returns
        -------

        Either a dict formatted appropriately for assignment as the **inputs** argument of the Composition's `run()
        method (form = *DICT*, the default), or string showing the format required by the **inputs** argument
        <Composition.run>` (form = *TEXT*).

        """

        if alias:
            warnings.warn(f"{alias} is aliased to get_input_format(); please use that in the future.")

        def _get_labels(labels_dict, index, input_port):
            """Need index for InputPort, since owner Mechanism is not passed in."""

            try:
                return list(labels_dict[input_port.name].keys())
            except KeyError:
                try:
                    return list(labels_dict[index].keys())
                except KeyError:
                    # Dict with no InputPort-specific subdicts, used to specify labels for all InputPorts of Mechanism
                    return list(labels_dict.keys())
            except:
                assert False, f"PROGRAM ERROR: Unable to find labels for " \
                              f"'{input_port.full_name}' of '{input_port.owner.name}'."

        def _get_inputs(comp, nesting_level=1, use_labels=False, template_dict=str):

            format_description_string = ''
            indent = '\t' * nesting_level
            template_input_dict = {}

            for node in comp.get_nodes_by_role(NodeRole.INPUT):
                node_inputs_for_format_string = []
                format_description_string += '\n' + indent + node.name + ': '
                node_inputs_for_template_dict = []
                node_key = node.name if use_names else node

                # Nested Compositions
                if show_nested_input_nodes and isinstance(node, Composition):
                    # No need for node_inputs_for_template_dict here as template_dict never contains nested_input_nodes
                    node_inputs_for_format_string = _get_inputs(node,
                                                                nesting_level=nesting_level + 1,
                                                                use_labels=use_labels)

                else:
                    for t in range(num_trials):
                        inputs_for_format = []
                        inputs_for_template_dict = []

                        # Mechanism with labels
                        if use_labels and isinstance(node, Mechanism) and node.input_labels_dict:
                            labels_dict = node.input_labels_dict

                            for i in range(len(node.external_input_shape)):
                                labels = _get_labels(labels_dict, i, node.input_ports[i])
                                inputs_for_format.append(repr(labels[t % len(labels)]))
                                inputs_for_template_dict.append(labels[t % len(labels)])
                            trial = f"[{','.join(inputs_for_format)}]"

                        # Mechanism(s) with labels in nested Compositions
                        elif (use_labels and isinstance(node, Composition)
                              and any(n.input_labels_dict for n
                                      in node._get_nested_nodes_with_same_roles_at_all_levels(node, NodeRole.INPUT))):

                            for port in node.input_CIM.input_ports:
                                input_port, mech, __ = node.input_CIM._get_destination_info_from_input_CIM(port)
                                labels_dict = mech.input_labels_dict
                                if labels_dict:
                                    labels = _get_labels(labels_dict, mech.input_ports.index(input_port), input_port)
                                    inputs_for_format.append(repr([labels[t % len(labels)]]))
                                    inputs_for_template_dict.append([labels[t % len(labels)]])
                                else:
                                    inputs_for_template_dict.append(port.default_input_shape)
                                    inputs_for_format.append(repr(np.array(port.default_input_shape).tolist()))
                            trial = f"[{','.join(inputs_for_format)}]"

                        # No Mechanism(s) with labels or use_labels == False
                        else:
                            inputs_for_template_dict = [port.default_input_shape for port in node.external_input_ports]
                            trial = f"[{','.join([repr(i.tolist()) for i in inputs_for_template_dict])}]"

                        node_inputs_for_format_string.append(trial)
                        node_inputs_for_template_dict.append(inputs_for_template_dict)

                    node_inputs_for_format_string = ', '.join(node_inputs_for_format_string)
                    if num_trials > 1:
                        node_inputs_for_format_string = f"[ {node_inputs_for_format_string} ]"

                format_description_string += node_inputs_for_format_string
                if not show_nested_input_nodes:
                    format_description_string += ','
                template_input_dict[node_key]=node_inputs_for_template_dict

            nesting_level -= 1
            if form == DICT:
                return template_input_dict
            else:
                return format_description_string

        if num_trials == FULL:
            num_trials = 1
            # Get number of labels in largest list of any input_labels_dict in an INPUT Mechanism
            for node in self._get_nested_nodes_with_same_roles_at_all_levels(self, NodeRole.INPUT):
                if node.input_labels_dict:
                    labels_dict = node.input_labels_dict
                    num_trials = max(num_trials, max([len(labels) for labels in labels_dict.values()]))

        # Return dict usable for run()
        if form == DICT:
            show_nested_input_nodes = False
            return _get_inputs(self, 1, use_labels, form)
        # Return text format
        else:
            formatted_input = _get_inputs(self, 1, use_labels)
            if show_nested_input_nodes:
                plural = 's' if num_trials > 1 else ''
                preface = f"\nInputs to (nested) INPUT Nodes of {self.name} for {num_trials} trial{plural}:"
                epilog = f"\n\nFormat as follows for inputs to run():\n" \
                         f"{self.get_input_format(form=form, num_trials=num_trials)}"
                return preface + formatted_input[:-1] + epilog
            return '{' + formatted_input[:-1] + '\n}'

    # Aliases for get_results_by_node:
    def get_output_format(self, **kwargs):
        return self.get_results_by_nodes(**kwargs, alias="get_output_format")

    def get_result_format(self, **kwargs):
        return self.get_results_by_nodes(**kwargs, alias="get_result_format")

    def get_results_format(self, **kwargs):
        return self.get_results_by_nodes(**kwargs, alias="get_results_format")

    def get_results_for_node(self, **kwargs):
        return self.get_results_by_nodes(**kwargs, alias="get_results_for_node")

    def get_results_for_nodes(self, **kwargs):
        return self.get_results_by_nodes(**kwargs, alias="get_results_for_nodes")

    def get_results_by_node(self, **kwargs):
        return self.get_results_by_nodes(**kwargs, alias="get_results_by_node")

    def get_results_by_nodes(self,
                             nodes:Union[Mechanism, list]=None,
                             use_names:bool=False,
                             use_labels:bool=False,
                             alias:str=None):
        """Return ordered dict with origin Node and current value of each item in results.

        .. note::
           Items are listed in the order of their values in the Composition's `results <Composition.results>` attribute,
           irrespective of the order in which they appear in the **nodes** argument if specified.

        Arguments
        ---------

        nodes : List[Mechanism or str], Mechanism or str : default None
            specifies `Nodes <Composition_Nodes>` for which to report the value; can be a reference to a Mechanism,
            its name, or a list of either or both.  If None (the default), the `values <Mechanism_Base.value>` of
            all `OUTPUT <NodeRole.OUTPUT>` Nodes are reported.

        use_names : bool : default False
            specifies whether to use the names of `Nodes <Composition_Nodes>` rather than references to them as keys.

        use_labels : bool : default False
            specifies whether to use labels to report the `values <Mechanism_Base.value>` of Nodes for `Mechanisms
            Mechanism` that have an `output_labels_dict <Mechanism_Base.output_labels_dict>` attribute.

        Returns
        -------

        Node output_values : Dict[Mechanism:value]
            dict , the keys of which are either Mechanisms or the names of them, and values are their
            `output_values <Mechanism_Base.output_values>`.
        """

        if alias:
            warnings.warn(f"{alias} is aliased to get_results_by_nodes(); please use that in the future.")

        # Get all OUTPUT Nodes in (nested) Composition(s)
        output_nodes = [self.output_CIM._get_source_info_from_output_CIM(port)[1]
                        for port in self.output_CIM.output_ports]

        # Get all values for all OUTPUT Nodes
        if use_labels:
            # Get labels for corresponding values
            values = [node.labeled_output_values for node in output_nodes]
        else:
            values = self.results[-1] or self.output_values

        full_output_set = zip(output_nodes, values)

        nodes = convert_to_list(nodes)
        # Translate any Node names to object references
        if nodes:
            bad_nodes = []
            for i, node in enumerate(nodes.copy()):
                if node in output_nodes:
                    continue
                if isinstance(node, str):
                    nodes[i] = next((n for n in output_nodes if n.name == node),None)
                    if nodes[i]:
                        continue
                bad_nodes.append(node)
                raise CompositionError(f"Nodes specified in get_results_by_nodes() method not found in {self.name} "
                                       f"nor any Compositions nested within it: {bad_nodes}")

        # Use nodes if specified, else all OUTPUT Nodes
        nodes = nodes or output_nodes
        # Get Nodes and values for ones specified in Nodes (all by default)
        result_set = [(n,v) for n, v in full_output_set if n in nodes]

        if use_names:
            # Use names of Nodes
            return {k.name:np.array(v).tolist() for k,v in result_set}
        else:
            return {k:np.array(v).tolist() for k,v in result_set}

    def _update_learning_parameters(self, context):
        pass

    @handle_external_context(fallback_most_recent=True)
    def reset(self, values=None, include_unspecified_nodes=True, context=NotImplemented):
        if not values:
            values = {}

        for node in self.stateful_nodes:
            if not include_unspecified_nodes and node not in values:
                continue
            reset_val = values.get(node)
            node.reset(reset_val, context=context)

    @handle_external_context(fallback_most_recent=True)
    def initialize(self, values=None, include_unspecified_nodes=True, context=None):
        """
            Initializes the values of nodes within cycles. If `include_unspecified_nodes` is True and a value is
            provided for a given node, the node will be initialized to that value. If `include_unspecified_nodes` is
            True and a value is not provided, the node will be initialized to its default value. If
            `include_unspecified_nodes` is False, then all nodes must have corresponding initialization values. The
            `DEFAULT` keyword can be used in lieu of a numerical value to reset a node's value to its default.

            If a context is not provided, the most recent context under which the Composition has executed will be used.

            Arguments
            ----------
            values: Dict { Node: Node Value }
                A dictionary contaning key-value pairs of Nodes and initialization values. Nodes within cycles that are
                not included in this dict will be initialized to their default values.

            include_unspecified_nodes: bool
                Specifies whether all nodes within cycles should be initialized or only ones specified in the provided
                values dictionary.

            context: Context
                The context under which the nodes should be initialized. context will be set to
                self.most_recent_execution_context if one is not specified.

        """
        # comp must be initialized from context before cycle values are initialized
        self._initialize_from_context(context, override=False)

        if not values:
            values = {}

        cycle_nodes = set(self.get_nodes_by_role(NodeRole.CYCLE) + self.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))

        for node in values:
            if node not in self.nodes:
                raise CompositionError(f"{node.name} "
                                       f"(entry in initialize values arg) is not a node in '{self.name}'")
            if node not in cycle_nodes:
                warnings.warn(
                    f"A value is specified for {node.name} of {self.name} in the 'initialize_cycle_values' "
                    f"argument of call to run, but it is neither part of a cycle nor a FEEDBACK_SENDER. "
                    f"Its value will be overwritten when the node first executes, and therefore not used."
                )

        for node in cycle_nodes:
            if not include_unspecified_nodes:
                if node not in values:
                    continue
            provided_value = values.get(node)
            value = provided_value if not provided_value == DEFAULT else node.defaults.value
            node.initialize(value, context)

    def disable_all_history(self):
        """
            When run, disables history tracking for all Parameters of all Components used in the Composition
        """
        self._set_all_parameter_properties_recursively(history_max_length=0)

    def _get_processing_condition_set(self, node):
        dep_group = []
        for group in self.scheduler.consideration_queue:
            if node in group:
                break
            dep_group = group

        # This condition is used to check of the step has passed.
        # Not all nodes in the previous step need to execute
        # (they might have other conditions), but if any one does we're good
        # FIXME: This will fail if none of the previously considered
        # nodes executes in this pass, but that is unlikely.
        conds = [Any(*(AllHaveRun(dep, time_scale=TimeScale.PASS) for dep in dep_group))] if len(dep_group) else []
        if node in self.scheduler.conditions:
            conds.append(self.scheduler.conditions[node])

        return All(*conds)

    def _input_matches_variable(self, input_value, var):
        var_shape = convert_to_np_array(var).shape
        # input_value ports are uniform
        if convert_to_np_array(input_value, dimension=2).shape == var_shape:
            return "homogeneous"
        # input_value ports have different lengths
        elif len(var_shape) == 1 and isinstance(var[0], (list, np.ndarray)):
            for i in range(len(input_value)):
                if len(input_value[i]) != len(var[i]):
                    return False
            return "heterogeneous"
        return False

    def _is_learning(self, context):
        """Returns true if the composition can learn in the given context"""
        return (not self.disable_learning) and (ContextFlags.LEARNING_MODE in context.runmode)

    def _build_variable_for_input_CIM(self, inputs):
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

        return build_CIM_input

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

    def _parse_runtime_params_conditions(self, runtime_params):
        """Validate runtime_params and assign Always() for any params that don't have a Condition already specified.
        Recursively process subdicts (Port- or Project-specific dictionaries of params).
        """
        def validate_and_assign_default_condition(node, entry, param_key, param_value):
            if not isinstance(param_value, tuple):
                param_spec = param_value
                # Default Condition
                param_condition = Always()
            # Parameter specified as tuple
            else:
                param_spec = param_value[0]
                if len(param_value)==1:
                    # Default Condition
                    param_condition = Always()
                elif len(param_value)==2:
                    # Condition specified, so use it
                    param_condition = param_value[1]
                else:
                    # Invalid tuple
                    raise CompositionError(f"Invalid runtime parameter specification "
                                           f"for {node.name}'s {param_key} parameter in {self.name}: "
                                           f"'{entry}: {param_value}'. "
                                           f"Must be a tuple of the form (parameter value, condition), "
                                           f"or simply the parameter value.")
            if isinstance(param_spec, dict):
                for entry in param_spec:
                    param_spec[entry] = validate_and_assign_default_condition(node,
                                                                              entry,
                                                                              param_key,
                                                                              param_spec[entry])
            return (param_spec, param_condition)

        if runtime_params is None:
            return {}
        for node in runtime_params:
            for param in runtime_params[node]:
                param_value = runtime_params[node][param]
                runtime_params[node][param] = validate_and_assign_default_condition(node,
                                                                                    runtime_params[node],
                                                                                    param,
                                                                                    param_value)
        return runtime_params

    def _get_satisfied_runtime_param_values(self, runtime_params, scheduler, context):
        """Return dict with values for all runtime_params the Conditions of which are currently satisfied.
        Recursively parse nested dictionaries for which Condition on dict is satisfied.
        """

        def get_satisfied_param_val(param_tuple):
            """Return param value if Condition is satisfied, else None."""
            param_val, param_condition = param_tuple
            if isinstance(param_val, dict):
                execution_params = parse_params_dict(param_val)
                if execution_params:
                    param_val = execution_params
                else:
                    return None
            if param_condition.is_satisfied(scheduler=scheduler,context=context):
                return param_val
            else:
                return None

        def parse_params_dict(params_dict):
            """Return dict with param:value entries only for Conditions that are satisfied."""
            execution_params = {}
            for entry in params_dict:
                execution_param = get_satisfied_param_val(params_dict[entry])
                if execution_param is None:
                    continue
                execution_params[entry] = execution_param
            return execution_params

        return parse_params_dict(runtime_params)

    def _after_agent_rep_execution(self, context=None):
        pass

    def _update_default_variable(self, *args, **kwargs):
        # NOTE: Composition should not really have a default_variable,
        # but does as a result of subclassing from Component.
        # Subclassing may not be necessary anymore
        raise TypeError(f'_update_default_variable unsupported for {self.__class__.__name__}')

    def _get_parsed_variable(self, *args, **kwargs):
        raise TypeError(f'_get_parsed_variable unsupported for {self.__class__.__name__}')

    def _delete_contexts(self, *contexts, check_simulation_storage=False, visited=None):
        super()._delete_contexts(*contexts, check_simulation_storage=check_simulation_storage, visited=visited)

        for c in contexts:
            try:
                self.scheduler._delete_counts(c.execution_id)
            except AttributeError:
                self.scheduler._delete_counts(c)

    def _initialize_as_agent_rep(self, context, base_context, alt_controller=None):
        assert self.controller is None or alt_controller is None

        _initialized = set()  # avoid reinitializing shared dependencies below
        self._initialize_from_context(
            context, base_context=base_context, override=True, visited=_initialized
        )
        if alt_controller is not None:
            # evaluation will be done with a controller from another composition
            alt_controller._initialize_from_context(
                context, base_context=base_context, override=True, visited=_initialized
            )

    def _clean_up_as_agent_rep(self, context, alt_controller=None):
        _deleted = set()  # avoid traversing shared dependencies below
        self._delete_contexts(context, visited=_deleted, check_simulation_storage=True)
        if alt_controller is not None:
            alt_controller._delete_contexts(
                context, visited=_deleted, check_simulation_storage=True
            )

    # endregion EXECUTION

    # ******************************************************************************************************************
    # region -------------------------------------- LLVM ---------------------------------------------------------------
    # ******************************************************************************************************************

    @property
    def _inner_projections(self):
        # PNL considers afferent projections to input_CIM to be part
        # of the nested composition. Filter them out.
        return (p for p in self.projections
                  if p.receiver.owner is not self.input_CIM and
                     p.receiver.owner is not self.parameter_CIM and
                     p.sender.owner is not self.output_CIM)

    def _get_param_ids(self):
        return ["nodes", "projections"] + super()._get_param_ids()

    def _get_param_struct_type(self, ctx):
        node_param_type_list = (ctx.get_param_struct_type(m) for m in self._all_nodes)
        proj_param_type_list = (ctx.get_param_struct_type(p) for p in self._inner_projections)
        comp_param_type_list = ctx.get_param_struct_type(super())

        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(node_param_type_list),
            pnlvm.ir.LiteralStructType(proj_param_type_list),
            *comp_param_type_list))

    def _get_state_ids(self):
        return ["nodes", "projections"] + super()._get_state_ids()

    def _get_state_struct_type(self, ctx):
        node_state_type_list = (ctx.get_state_struct_type(m) for m in self._all_nodes)
        proj_state_type_list = (ctx.get_state_struct_type(p) for p in self._inner_projections)
        comp_state_type_list = ctx.get_state_struct_type(super())

        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(node_state_type_list),
            pnlvm.ir.LiteralStructType(proj_state_type_list),
            *comp_state_type_list))

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
        comp_states = super()._get_state_initializer(context)

        return (tuple(node_states), tuple(proj_states), *comp_states)

    def _get_param_initializer(self, context):
        node_params = (m._get_param_initializer(context) for m in self._all_nodes)
        proj_params = (p._get_param_initializer(context) for p in self._inner_projections)
        comp_params = super()._get_param_initializer(context)

        return (tuple(node_params), tuple(proj_params), *comp_params)

    def _get_data_initializer(self, context):
        output_data = ((os.parameters.value.get(context) for os in m.output_ports) for m in self._all_nodes)
        nested_data = (getattr(node, '_get_data_initializer', lambda _: ())(context)
                       for node in self._all_nodes)
        return (pnlvm._tupleize(output_data), *nested_data)

    def _get_node_index(self, node):
        node_list = list(self._all_nodes)
        return node_list.index(node)

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "run" in tags:
            return pnlvm.codegen.gen_composition_run(ctx, self, tags=tags)
        else:
            return pnlvm.codegen.gen_composition_exec(ctx, self, tags=tags)

    def enable_logging(self):
        for item in self.nodes + self.projections:
            if isinstance(item, Composition):
                item.enable_logging()
            elif not isinstance(item, CompositionInterfaceMechanism):
                for param in item.parameters:
                    if param.loggable and param.log_condition is LogCondition.OFF:
                        param.log_condition = LogCondition.EXECUTION

    # endregion LLVM

    def as_mdf_model(self, simple_edge_format=True):
        """Creates a ModECI MDF Model representing this Composition

        Args:
            simple_edge_format (bool, optional): If True, Projections
            with non-identity matrices are constructed as . Defaults to True.

        Returns:
            modeci_mdf.Model: a ModECI Model representing this Composition
        """
        import modeci_mdf.mdf as mdf

        def is_included_projection(proj):
            included_types = (
                CompositionInterfaceMechanism,
                LearningMechanism,
                OptimizationControlMechanism,
            )
            return (
                not isinstance(proj.sender.owner, included_types)
                and not isinstance(proj.receiver.owner, included_types)
                and not isinstance(proj, (AutoAssociativeProjection, ControlProjection))
            )
        nodes_dict = {}
        projections_dict = {}
        self_identifier = parse_valid_identifier(self.name)
        metadata = self._mdf_metadata

        additional_projections = []
        additional_nodes = (
            [self.controller]
            if self.controller is not None
            else []
        )

        for n in list(self.nodes) + additional_nodes:
            if not isinstance(n, CompositionInterfaceMechanism):
                nodes_dict[parse_valid_identifier(n.name)] = n.as_mdf_model()

            # consider making this more general in the future
            try:
                additional_projections.extend(n.control_projections)
            except AttributeError:
                pass

        for p in list(self.projections) + additional_projections:
            # filter projections to/from CIMs of this composition
            # and projections to things outside this composition
            if is_included_projection(p):
                try:
                    pre_edge, edge_node, post_edge = p.as_mdf_model(simple_edge_format)
                except TypeError:
                    edges = [p.as_mdf_model(simple_edge_format)]
                else:
                    nodes_dict[edge_node.id] = edge_node
                    edges = [pre_edge, post_edge]
                    if 'excluded_node_roles' not in metadata[MODEL_SPEC_ID_METADATA]:
                        metadata[MODEL_SPEC_ID_METADATA]['excluded_node_roles'] = []

                    metadata[MODEL_SPEC_ID_METADATA]['excluded_node_roles'].append([edge_node.id, str(NodeRole.OUTPUT)])

                for e in edges:
                    projections_dict[e.id] = e

        metadata[MODEL_SPEC_ID_METADATA]['controller'] = self.controller.as_mdf_model() if self.controller is not None else None

        graph = mdf.Graph(
            id=self_identifier,
            conditions=self.scheduler.as_mdf_model(),
            **self._mdf_model_parameters[self._model_spec_id_parameters],
            **metadata
        )

        for _, node in nodes_dict.items():
            graph.nodes.append(node)

        for _, proj in projections_dict.items():
            graph.edges.append(proj)

        return graph

    # ******************************************************************************************************************
    # region ----------------------------------- PROPERTIES ------------------------------------------------------------
    # ******************************************************************************************************************

    @property
    def input_ports(self):
        """Return all InputPorts that belong to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_ports

    @property
    def input_port(self):
        """Return the index 0 InputPort that belongs to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_ports[0]

    @property
    def input_values(self):
        """Return values of all InputPorts that belong to the Input CompositionInterfaceMechanism"""
        return self.get_input_values()

    def get_input_values(self, context=None):
        return [input_port.parameters.value.get(context) for input_port in self.input_CIM.input_ports]

    @property
    def output_port(self):
        """Return the index 0 OutputPort that belongs to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_ports[0]

    @property
    def output_ports(self):
        """Return all OutputPorts that belong to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_ports

    @property
    def output_values(self):
        """Return values of all OutputPorts that belong to the Output CompositionInterfaceMechanism in the most recently executed context"""
        return self.get_output_values(self.most_recent_context)

    def get_output_values(self, context=None):
        return [output_port.parameters.value.get(context)
                for output_port in self.output_CIM.output_ports
                if (not self.output_CIM._sender_is_probe(output_port) or self.include_probes_in_output)]

    @property
    def shadowing_dict(self):
        """Return dict with shadowing ports as the keys and the ports they shadow as values."""
        return {port:port.shadow_inputs for node in self._all_nodes for port in node.input_ports if port.shadow_inputs}

    @property
    def mechanisms(self):
        return MechanismList(self, [node if isinstance(node, Mechanism) else node.mechanisms
                                    for node in self.nodes])

    @property
    def runs_simulations(self):
        return True

    @property
    def simulation_results(self):
        return self.parameters.simulation_results.get(self.default_execution_id)

    @property
    def external_input_ports(self):
        """Return all external InputPorts that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_port for input_port in self.input_CIM.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_ports_of_all_input_nodes(self):
        """Return all external InputPorts of all INPUT Nodes (including nested ones) of Composition.
        Note: the InputPorts returned are those of the actual Mechanisms
              to which the ones returned by external_input_ports ultimately project.
        """
        try:
            # return [self._get_destination(output_port.efferents[0])[0]
            #         for _,output_port in self.input_CIM.port_map.values()]
            return self._get_input_receivers(comp=self, type=PORT, comp_as_node=False)
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_shape(self):
        """Alias for _default_external_input_shape"""
        return self._default_external_input_shape

    @property
    def _default_external_input_shape(self):
        """Return default_input_shape of all external InputPorts that belong to Input CompositionInterfaceMechanism"""
        try:
            return [input_port.default_input_shape for input_port in self.input_CIM.input_ports
                    # FIX: 2/4/22 - IS THIS NEEDED (HERE OR BELOW -- DO input_CIM.input_ports EVER GET ASSIGNED THIS?
                    if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_variables(self):
        """Return variables of all external InputPorts that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_port.parameters.variable.get()
                    for input_port in self.input_CIM.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def default_external_input_variables(self):
        """Return default variables of all external InputPorts that belong to the Input CompositionInterfaceMechanism
        """

        try:
            return [input_port.defaults.variable for input_port in self.input_CIM.input_ports if
                    not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_values(self):
        """Return values of all external InputPorts that belong to the Input CompositionInterfaceMechanism"""
        try:
            #  FIX: 2/4/22 SHOULD input_port.variable REPLACE input_port.value HERE?
            return [input_port.parameters.value.get()
                    for input_port in self.input_CIM.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def default_external_input_values(self):
        """Return the default values of all external InputPorts that belong to the Input CompositionInterfaceMechanism
        """

        try:
            return [input_port.defaults.value for input_port in self.input_CIM.input_ports if
                    not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def stateful_nodes(self):
        """
        List of all nodes in the Composition that are currently marked as stateful. For Mechanisms, statefulness is
        determined by checking whether node.has_initializers is True. For Compositions, statefulness is determined
        by checking whether any of its `Nodes <Composition_Nodes>` are stateful.

        Returns
        -------
        all stateful nodes in the `Composition` : List[`Node <Composition_Nodes>`]

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
    def class_parameters(self):
        return self.__class__.parameters

    @property
    def stateful_parameters(self):
        return [param for param in self.parameters if param.stateful]

    @property
    def random_variables(self):
        """Return list of Components with seed Parameters (i.e., ones that that call a random function)."""
        return [param._owner._owner for param in self.all_dependent_parameters('seed').keys()]

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
        return [node for node in self.nodes if NodeRole.LEARNING in self.nodes_to_roles[node]]

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
    def feedback_senders(self):
        return ContentAddressableList(component_type=Component,
                                      list=[node for node in self.nodes
                                            if node in self.get_nodes_by_role(NodeRole.FEEDBACK_SENDER)])

    @property
    def feedback_receivers(self):
        return ContentAddressableList(component_type=Component,
                                      list=[node for node in self.nodes
                                            if node in self.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER)])

    @property
    def feedback_projections(self):
        return ContentAddressableList(component_type=Projection,
                                      list=[p.component for p in self.graph.vertices
                                            if p.feedback is EdgeType.FEEDBACK])

    @property
    def _all_nodes(self):
        for n in self.nodes:
            yield n
        yield self.input_CIM
        yield self.output_CIM
        yield self.parameter_CIM
        if self.controller:
            yield self.controller

    # endregion PROPERTIES

    # ******************************************************************************************************************
    # region ----------------------------------- SHOW_GRAPH ------------------------------------------------------------
    # ******************************************************************************************************************

    def show_graph(self,
                   show_all=False,
                   show_node_structure=False,
                   show_nested=NESTED,
                   show_nested_args=ALL,
                   show_cim=False,
                   show_controller=True,
                   show_learning=False,
                   show_headers=True,
                   show_types=False,
                   show_dimensions=False,
                   show_projection_labels=False,
                   show_projections_not_in_composition=False,
                   active_items=None,
                   output_fmt='pdf',
                   context=None):

        return self._show_graph(show_all=show_all,
                                show_node_structure=show_node_structure,
                                show_nested=show_nested,
                                show_nested_args=show_nested_args,
                                show_cim=show_cim,
                                show_controller=show_controller,
                                show_learning=show_learning,
                                show_headers=show_headers,
                                show_types=show_types,
                                show_dimensions=show_dimensions,
                                show_projection_labels=show_projection_labels,
                                show_projections_not_in_composition=show_projections_not_in_composition,
                                active_items=active_items,
                                output_fmt=output_fmt,
                                context=context)

    def _set_up_animation(self, context):
        self._show_graph._set_up_animation(context)

    def _animate_execution(self, active_items, context):
        self._show_graph._animate_execution(active_items, context)

    # endregion SHOW_GRAPH

    def make_likelihood_function(self, *args, **kwargs):
        """
        This method invokes :func:`~psyneulink.core.components.functions.fitfunctions.make_likelihood_function`
        on the composition.
        """
        return make_likelihood_function(composition=self, *args, **kwargs)


def get_compositions():
    """Return list of Compositions in caller's namespace."""
    frame = inspect.currentframe()
    return [c for c in frame.f_back.f_locals.values() if isinstance(c, Composition)]
