# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* AutodiffComposition *************************************************

"""

Contents
--------

  * `AutodiffComposition_Overview`
  * `AutodiffComposition_Creation`
      COMMENT:
      - `AutodiffComposition_Optimizer`
      COMMENT
      - `AutodiffComposition_Learning_Pathways`
          - `AutodiffComposition_Sample`
          - `AutodiffComposition_Target`
          - `AutodiffComposition_LossMechanism`
          - `AutodiffComposition_Specifying_Learning_Pathways`
      - `AutodiffComposition_Learning_Rates`
      - `AutodiffComposition_Exchange_With_Torch_Parameters`
      - `AutodiffComposition_Restrictions`
  * `AutodiffComposition_Structure`
      - `AutodiffComposition_Learning_Components`
          - `AutodiffComposition_Structure_LossMechanisms`
          - `AutodiffComposition_Structure_Target_Nodes`
      - `AutodiffComposition_PytorchRepresentation`
      - `AutodiffComposition_Nesting`
  * `AutodiffComposition_Execution`
      - `AutodiffComposition_PyTorch`
          - `AutodiffComposition_Additional_Optimization_Steps`
          COMMENT:
          - `AutodiffComposition_Exclusion_From_Gradient_Calculation`
          COMMENT
          - `AutodiffComposition_Synchronization_with_PyTorch`
          - `AutodiffComposition_Save_Pytorch_Training_Data`
      - `AutodiffComposition_LLVM`
      - `AutodiffComposition_Python`
      - `AutodiffComposition_Logging`
  * `AutodiffComposition_Examples`
  * `AutodiffComposition_Class_Reference`


.. _AutodiffComposition_Overview:

Overview
--------

AutodiffComposition is a subclass of `Composition` for constructing and training neural networks using
`PyTorch <https://pytorch.org/>`_ and, in some cases, direct compilation using `LLVM <AutodiffComposition_LLVM>`.
These can considerably accelerate training, by as much as three orders of magnitude compared to `Python mode
<Composition_Learning_Standard>` used by a standard Composition. An AutodiffComposition is constructed and
executed in the same way as a standard Composition, though it provides additional `functionality
<Composition_Compilation_Table>`, including:

.. _AutodiffComposition_Additional_Functionality:

  - use of `internal target signals <AutodiffComposition_Target>` for training;
  - training of `nested Compositions <AutodiffComposition_Nesting>`.
  - training of recurrent neural networks (RNNs, e.g., `GRUComposition`);
  - training of external (episodic) memory structures (e.g., `EMComposition`);

In addition to supporting `supervised learning <Composition_Learning_Supervised>` using the `backpropagation learning
algorithm <https://en.wikipedia.org/wiki/Backpropagation>`_, it also supports some forms of `unsupervised learning
<Composition_Learning_Unsupervised>` that are possible in PyTorch (e.g., `self-organized maps
<https://github.com/giannisnik/som>`_).

.. _AutodiffComposition_Creation:

Creating an AutodiffComposition
-------------------------------

An AutodiffComposition is created in the same way as a standard Composition, with the following differences:

- learning pathways are configured by specifing pairs of `samples <AutodiffComposition_Sample>` (or
  "students") and `targets <AutodiffComposition_Target>` (or "teachers"), each of which is a `Mechanism`
  or the `OutputPort` of one, and the values of which are used to compute the loss on each trial of training
  (see `AutodiffComposition_Learning_Pathways` for details of specification);

- the constructor includes a number of `additional arguments <AutodiffComposition_Configuring_Learning>`
  that are specific to the AutodiffComposition;

- there are some `restrictions <AutodiffComposition_Restrictions>` that apply to its construction;

- an AutodiffComposition's `pytorch_representation <AutodiffComposition.pytorch_representation>` is used to
  execute it in PyTorch, which is constructed when its `learn() <Composition.learn>` method is called
  (see `AutodiffComposition_PytorchRepresentation` for additional details).

COMMENT:
BREADCRUMB - UPDATE ONCE KATHERINE'S CHANGES HAVE BEEN INCORPORATED
.. _AutodiffComposition_Optimizer:

*Optimizer*
~~~~~~~~~~~

In addition to `learning_rate <Projection.learning_rate>`, other parameters can be customized by constructing
a `torch.nn.optimizer <https://pytorch.org/docs/main/optim.html>`_ and assigining it to the **optimizer** argument
of either the AutodiffComposition's constructor or `learn <Composition.learn>` method.  This requires creating
and adding ``param_groups`` for the `torch.nn.Parameters
<https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>`_ corresponding to the Projections to be
specified, which are listed in the AutodiffComposition's `torch_parameters <AutodiffComposition.torch_parameters>`
attribute.
COMMENT

.. _AutodiffComposition_Learning_Pathways:

COMMENT:
TEACHER_TARGET BREADCRUMB: - FINISH WORDING THIS
                           - REVISE WHEN NO LEARNABLE PATHWAYS BECOMES AN ERROR RATHER THAN A WARNING
                           - MOVE OR COPY THIS TO SECTION ON STRUCDTURE BELOW
Because learning in implemeted by changes to `matrix <MappingProjection.matrix>` Parameters of `MappingProjections`,
only pathways that have at least one learnable MappingProjection (i.e., one with its `learnable <MappingProjection.learnable>`
attribute set to ``True``) can be used for learning.  Thus, only `samples <AutodiffComposition_Sample>` that are the
endpoints of such pathways can be specified in the **targets** argument of the AutodiffComposition's constructor,
and only those will be trained when the `learn() <Composition.learn>` method is called.
Pathways no learnable MappingProjections are ignored, and a warning is issued if any are specified in the
**targets** argument of the AutodiffComposition's constructor.  This includes the case in which there is a single
Mechanism in the pathway (i.e., it is a `SINGELTON <NodeRole.SINGLETON>`), since there are no Projections in such a pathway.

.. note::
   This differs from configuration in Pytorch, in which a single torch.nn.Module can be trained since it is
   automatically assigned parameters (based on its input dimensionality) at construction; this can be thought of as
   equivalent to -- and can be replicated in PsyNeulink by -- construting a pathway with a single MappingProjection
   from an input Node to a Mechanism that that corresponds to (i.e., implements the same function os) the
   torch.nn.Module being trained.  In other words, in PsyNeulink, the equivalent of a module's parameters must be
   constructed explicity in the form of an afferent `MappingProjection` which, in turn, requires a node that sends
   that Projection to the Mechanism.

.. technical_note::
   The technical reason that a pathway with only a SINGLETON <NodeRole.SINGLETON> Node cannot be trained is that
   its afferent and efferent MappingProjections are from the `input_CIM <Composition.input_CIM>` and to the
   `output_CIM <Composition.output_CIM>` of the Composition to which it belongs. Such MappingProjectiosn
    (i.e., from an input_CIM to its INPUT Nodes nor those from its OUTPUT Nodes to its output_CIM) are not
   learnable; they serve simply as conduits of information between the Composition and either the Composition within
   which it is nested, or the "outside world."


COMMENT

*Configuring Learning Pathways*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each learning pathway is configured by specifying either a `sample <AutodiffComposition_Sample>`\\-`target
<AutodiffComposition_Target>` target pair, or a `LossMechanism` that uses these, in the **targets** argument of
the AutodiffComposition's constructor. These Components are described below, followed by the ways in which they
can be `specified <AutodiffComposition_Specifying_Learning_Pathways>` in the constructor's **targets** argument.

.. _AutodiffComposition_Sample:

*Sample*
^^^^^^^^
This is a Component that generates the value of a pathway being trained (sometimes referred to as a "student"); it is
specified by the `ProcessingMechanism` at the end of the `pathway <Composition_Pathways>`, or one of its `OutputPorts
<OutputPort>`. The *sample* can be anywhere in the AutodiffComposition, or in one `nested <AutodiffComposition_Nesting>`
within it. It is trained using the value of the `target <AutodiffComposition_Target>` with which it is paired, or by
values specified in the **targets** argument of the `learn() <Composition.learn>` method (see `below
<AutodiffComposition_Specifying_Learning_Pathways>`). Only one *target* can be associated with a *sample*,
though a target can be assigned to multiple *samples*.

    .. note::
       Although a *sample* can be assigned only one *target*, it can participate in (i.e., be an intermediate Node)
       in other learning pathways, in which case the error signal it receives from its *target* will be combined with
       those that are transmitted to it from any other learning pathways in which it participates when the gradients
       are calcuated by the AutodiffComposition's `backward <AutodiffComposition.autodiff_backward>` method.

.. _AutodiffComposition_Target:

*Target*
^^^^^^^^
This is the source of the value used to train the `sample <AutodiffComposition_Sample>`  with which it is paired
(sometimes referred to as a "teacher"); it can be the `value <OutputPort.value>` of another Mechanism in the
AutodiffComposition, or an external value value provided when the `learn() <Composition.learn>` method is
executed. An internal source can be any ProcessingMechanism in the AutodiffComposition or one `nested
<AutodiffComposition_Nesting>` within it, so long as it is not in the same `pathway <Composition_Pathways>` as the
sample it trains. This allows the value of one pathway to be used to train another. Alternatively, the kewyord
*TARGET* can be used to specify the *target* for a *sample*, which allows external values provided in the **targets**
argument of the `learn() <Composition.learn>` method to be used to train the pathway (see `below
<AutodiffComposition_Specifying_Learning_Pathways>`). In that case, a `TARGET Node
<AutodiffComposition_Structure_Target_Nodes>` is automtically constructed for the *sample*, to receive the external
input when learning is executed, and the values (assigned as inputs to the those *TARGET Nodes*) must be provided in
the **targets** argument of the `learn() <Composition.learn>` when it is called
(see `Autodiffcomposition_PyTorch`).

    .. hint::
      the same *target* can be used to train more than one *sample*.

    .. warning::
       If an internal source (i.e., a ProcessingMechanism) is specified for the *target* of a `sample
       <AutodiffComposition_Sample>` in the **targets** argument of the AutodiffComposition's constructor,
       then there should *NOT* be an entry for that *sample-target* pair in the **targets** argument of
       the `learn() <Composition.learn>` method; the presence of one will raise an error.

       Conversely, any *sample* paired with the keyword `TARGET` in the **targets** argument of
       the AutodiffComposition's constructor (specifying the use of external training signals)
       *MUST* appear in the **targets** argument of the `learn() <Composition.learn>`
       method, paired with one or more values to be used for training that sample during learning
       (see `Target Inputs <Composition_Target_Inputs>` for information specifying these).


.. _AutodiffComposition_LossMechanism:

*LossMechanism*
^^^^^^^^^^^^^^^

This calculates the loss for the current values of a `sample <AutodiffComposition_Sample>` and
`target <AutodiffComposition_Target>`. If the LossMechanism is specified explicity (see `below
<AutodiffComposition_Specifying_Learning_Pathways>`), it uses the form of `Loss` specified in
either the **loss** or **function** argument of its constructor; in this case then its *sample*
and *target* must also be specified in the corresponding arguments of the constructor. If a
LossMechanism is not specified explicity for a *sample-target* pair, one is automatically
constructed for them, and uses the `Loss` specified by the `loss_spec <AutodiffComposition.loss_spec>`
of the AutodiffComposition.

.. _AutodiffComposition_Specifying_Learning_Pathways:

*Specifying sample-target pairs*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is done in the **targets** argument of the AutodiffComposition's constructor, using any of the forms of
specificaiton listed below. If *any* *sample-target* pairs are specified, *only* those are used; if *none* are
specified, then all of `OUTPUT Nodes <Composition_Nodes>` of the AutodiffComposition are used as *samples*, and
corresponding `TARGET Nodes <AutodiffComposition_Structure_Target_Nodes>` are automatically constructed to receive
the target values specified for each in the **targets** argument of the AutodiffComposition's `learn()` method
when it is called.

    * *tuple*: (sample, <target or *TARGET*>);

    * *LossMechanism*: the **sample** and **target** arguments of its constructor must be specified; its **loss**
      argument can also be used to specify a form of `Loss`; if none is specified, then the loss is determined by
      the AutodiffComposition's `loss_spec <AutodiffComposition.loss_spec>` Parameter.

    * *list*: any combination of the above;

    * *dict*: {sample: <target or *TARGET*}; each entry specifies a *sample-target* pair.

  .. note::
     If `samples <AutodiffComposition_Sample>` and `targets <AutodiffComposition_Target>` are specified for *some*
     but *not* all of the learnable pathways (i.e. ones with `learnable <MappingProjection.learnable>` Projections)
     in the **targets** argument of the AutodiffComposition's constructor, a warning is issued indicating the learnable
     pathways that lack learning components (and, in particular, a `LossMechanism`), and for which learning will
     not occur.

COMMENT:
+============+==========================+==========================================================
|            |          constructor     |         learn()
+============+==========================+==========================================================
|            |                          |
|Composition |          N/A             |    all OUTPUT Nodes
|            |                          |    or none (assigned default_variables as target values)
|            |                          |
+============+==========================+==========================================================
|            |                          |
|            |  Any ProcessingMechanism |    All of the ones specified as TARGET in constructor
|            |                          |
| Autodiff   +------------------------- OR -------------------------------------------------------
|            |                          |
|            |          None            |    all OUTPUT Nodes
|            |                          |    or none (assigned default_variables as target values)
|            |                          |
|============+=====================================================================================
COMMENT

.. _AutodiffComposition_Learning_Rates:

*Learning Rates*
~~~~~~~~~~~~~~~~

The **learning** argument of the constructor and/or the `learn <Composition.learn>` method can be used to
specify a `learning_rate <Composition.learning_rate>` for an entire AutodiffComposition, ones nested within
it, and/or individual MappingProjections (see `Composition_Learning_rate` for details of specification, and the `table
<Composition_Learning_Rate_Precedence_Hierarchy>` for which specifications take prcedence over others). Learning_rates
specified for individual MappingProjections are passed to the corresponding parameters of the AutodiffComposition's
`pytorch_representation <AutodiffComposition.pytorch_representation>` when it is executed. Specifications made in the
constructor for the AutodiffComposition are used as the default learning_rates for all executions of the `learn
<Composition.learn>`; specifications made in the call to the `learn() <Composition.learn>` method
override any made in the constructor, but are used only for that execution. A warning is issued if a learning_rate is
specified for a Projection with a `learnable <MappingProjection.learnable>` attribute set to ``False``, and an error
is generated if the Projection is associated with a PyTorch Parameter that is not learnable.
See `Composition_Learning_rate` for additional information about specifying learning_rates, including how the
`learning_rate <MappingProjection.learning_rate>` is determined for Projections that are not expliclity specified.

.. hint::
   To disable learning for a particular `MappingProjection` in an AutodiffComposition, assign `False` either
   to the `learnable <MappingProjection.learnable>` argument in its constructor, or in an entry of a dict used
   to specify the **learning_rate** argument of the AutodiffComposition's constructor or its learn() method
   (see `Composition_Learning_rate`); this applies to MappingProjections at any level of `nesting
   <AutodiffComposition_Nesting>`.


.. _AutodiffComposition_Exchange_With_Torch_Parameters:

*Exchanging Parameters with Pytorch Modules*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AutodiffComposition's `copy_torch_param_to_projection_matrix` and `copy_projection_matrix_to_torch_param` methods
can be used to exchange weight matrices between the parameters of a PyTorch module and the `matrix
<MappingProjection.matrix>` Parameter of a `MappingProjection` in the AutodiffComposition. Pytorch Parameters can
be referenced either by the Parameter object itself, or by the module and either the name or index of the
Parameter in the module's state_dict or parameter list, respectively.Slices of PyTorch Parameters can also be used,
for cases in which the matrix of a Project corresponds to only a subpart of the PyTorch Parameter (e.g., for
`GRUComposition`). Both methods return the item assigned.

  .. warning::
     PsyNeuLink `matrix <MappingProjection.matrix>` Parameters are transposed with respect to PyTorch parameters.
     This is managed automatically by the copy methods noted above, but must be taken into account if either is
     accessed and/or copied to the other by any other means.

.. _AutodiffComposition_Restrictions:

*AutodiffComposition Restrictions*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _AutodiffComposition_Modulatory_Mechanisms:

.. _Autodiff_Control_Components:

*Control Components*. An AutodiffComposition can contain `ControlMechanisms <ControlMechanism>` or a `controller
<Composition_Controller>`, that will operate normally when it's `run() <Composition.run>`is run in both `Python mode
<AutodiffComposition_Python>` and `PyTorch mode <AutodiffComposition_PyTorch>`. However, at present, these are not
supported for learning in `PyTorch mode <AutodiffComposition_PyTorch>`; a warning is issued and these are ignored
when the `learn() <Composition.learn>` method is called with **execution_mode** = `ExecutionMode.PyTorch`.
Accomodation of control during learning in `PyTorch mode <AutodiffComposition_PyTorch>` will be implemented in a
future version.

.. _Autodiff_Learning_Components_Warning:

*PsyNeuLink Learning Components*.  An AutodiffComposition **cannot include any** `learning components
<Composition_Learning_Components>` themselves (i.e., `LearningMechanisms <LearningMechanism>`, `LearningSignals
<LearningSignal>`, or `LearningProjections <LearningProjection>`, nor the `ComparatorMechanism`
or `ObjectiveMechanism` used to compute the loss for learning). These are constructed
automatically when learning is executed in `Python mode <AutodiffComposition_Python>` or `LLVM mode
<AutodiffComposition_LLVM>`, and PyTorch-compatible Components are constructed when it is executed in
`PyTorch mode <AutodiffComposition_PyTorch>`.

.. _AutodiffComposition_Bias_Parameters:

*No Bias Parameters*. AutodiffComposition does not (currently) support the *automatic* construction of separate bias parameters.
Thus, when constructing the PyTorch version of an AutodiffComposition, the `bias
<https://www.pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ parameter of any PyTorch modules are set to False.
However, biases can be implemented using `Composition_Bias_Nodes`.

.. _AutodiffComposition_Post_Construction_Modification:

*No Post-construction Modification*. Mechanisms or Projections should not be added to or deleted from an
AutodiffComposition after it has been executed. Unlike an ordinary Composition, AutodiffComposition does not support
this functionality.

  .. technical_note::
     Post-construction modification is currently not possible because the `pytorch_representation
     <AutodiffComposition.pytorch_representation` is constructed at the time the AutodiffComposition is first
     constructed, and can't be modified after that.  This will be fixed in a future version.

.. _AutodiffComposition_Structure:

Structure
---------

.. _AutodiffComposition_Learning_Components:

*Learning Components*
~~~~~~~~~~~~~~~~~~~~~

The following learning components are constructed for an AutodiffComposition for use in `PyTorch mode
<AutodiffComposition_Pytorch>`, that are listed in its `learning_components <Composition.learning_components>`
attribute:

.. _AutodiffComposition_Structure_LossMechanisms:

*Loss Mechanisms*
^^^^^^^^^^^^^^^^^

This computes the loss for a given pathway, using its `sample <LossMechanism.sample>`, `target
<LossMechanism.target>`, and assigned form of `loss <LossMechanism.loss>`. It receives `MappingProjections
<MappingProjection>` from `sample <LossMechanism.sample>` and `target <LossMechanism.target>` Mechanisms,
each of which is non-learnable and assigned an `IDENTITY_MATRIX`. If the LossMechanism was generated
automatically (see `AutodiffComposition_Loss_Mechanism_Specification`), it uses the `loss_spec
<AutodiffComposition.loss_spec>` specified for the AutodiffComposition; if it was specified explicity, it
uses the form of `Loss` specified in the **loss** argument of its constructor, or the `PyTorch loss function
<https://pytorch.org/docs/stable/nn.html#loss-functions>`_ specified in the **function** argument of its constructor.

  .. technical_note::
     The `LossMechanism` of an AutodiffComposition is comparable to the `ComparatorMechanism` (of which it is
     a sublcass) used as the `OBJECTIVE MECHANISM <OBJECTIVE_MECHANISM>` to compute the error for learning in
     a standard Composition.

  .. technical_note::
     The tensor that the LossMechanism receives from its `target <LossMechanism.target>` is detached prior to
     its use in computing the loss, in order to prevent gradient propagation to the target Mechanism, which may
     be in its own `learning pathway <AutodiffComposition_Learning_Pathways>`.

.. _AutodiffComposition_Structure_Target_Nodes:

*TARGET Nodes*
^^^^^^^^^^^^^^

If any external `targets <AutodiffComposition_Target>` are specified in **targets** argument of the
AutodiffComposition's constructor, then a `TARGET Node <NodeRole.TARGET>` is constructed for the corresponding
`samples <AutodiffComposition_Sample>`.  If *no* *targets* are specified, then a *TARGET Node* is constructed for
every `OUTPUT Node <NodeRole.OUTPUT>` of the AutodiffComposition that belongs to a pathway with at least one
`learnable <MappingProjection.learnable>` Projection.  Each *TARGET Node* receives the input specified for its
corresponding *sample* in the **targets** argument of the `learn() <AutodiffComposition>` method, and projects to
the corresponding `LossMechanism`. The *TARGET Nodes* for an AutodiffComposition are listed in its `target_nodes
<AutodiffCompostion.target_nodes>` and `learning_components <Composition.learning_components>` attributes.

COMMENT:
TEACHER_TARGET BREADCRUMB: ADD NOTE HERE FROM ABOVE ABOUT SINGLETONS NOT BEING LEARNABLE
COMMENT

.. _AutodiffComposition_PytorchRepresentation:

*Pytorch Representation*
~~~~~~~~~~~~~~~~~~~~~~~~~

COMMENT:
BREADCRUMB:
  ; with each Projection assigned to a `torch parameter
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>`_
  , and its learning rate assigned to
  the corersponding parameters (see `AutodiffComposition_Learning_Rates` for specification of learning rates).
COMMENT


An AutodiffComposition uses a `pytorch_representation <AutodiffComposition.pytorch_representation>` to execute
learning when it's `learn() <Composition.learn>` method is called in `Pytorch mode
<AutodiffComposition_PyTorch>`.  This is comprised of a outer `PytorchCompositionWrapper` for the AutodiffComposition,
that is comprised of `PytorchMechanismWrappers <PytorchMechanismWrapper` and `PytorchProjectionWrappers
<PytorchProjectionWrappers>` for the Compositions Mechanisms and Projections, and `PytorchCompositionWrapper
<PytorchCompositionWrappers>` for any AutodCompositions that are nested within it. Although the `pytorch_representation
<AutodiffComposition.pytorch_representation>` maintains the hierarchical structure of any `nested Compositions
<Composition_Nested>`, when it is executed it "flattens" this, incorporating the nodes of any nested
AutodiffCompositions into the top level.  This can be shown graphically using the AutodiffComposition's
`show_graph <Composition.show_graph>` method, as described below.

The `pytorch_representation <AutodiffComposition.pytorch_representation>` is constructed automtically when the `learn()
<AutodiffComposition>` method of AutodiffComposition is executed in `PyTorch mode <AutodiffComposition_Pytorch>`
(the default), and is used to execute it in PyTorch. It is also constructed when the `show_graph <Composition.show_graph
method is called with its **show_pytorch** argument set to `True`, which generates a graphic display of the
`pytorch_representation <AutodiffComposition.pytorch_representation>`. As noted above, this shows the "flattened" verion
of the AutodiffComposition (if it has any `nested AutodiffCompositions <AutodiffComposition_Nesting>` within it) that
will execute in PyTorch, with direct Projections between Nodes at different levels of nesting. This also shows any
`LossMechanisms <AutodiffComposition_Structure_LossMechanisms>` and `TARGET Nodes
<AutodiffComposition_Structure_Target_Nodes>` that have been automatically constructed
(see `AutodiffComposition_LossMechanism` and `AutodiffComposition_Target`, respectively). Furthermore, note that `no
control-related components <Autodiff_Control_Components> are shown. Finally, Projections that are `excluded from
gradient calculations <PytorchMechanismWrapper.exclude_from_gradient_calc>` are shown with dotted arrows, which are also
used to show the flow of the training signal from the a `LossMechanism <AutodiffComposition_LossMechanism>` to the
`sample <AutodiffComposition_Sample>` for which it calculates the loss.

  .. note::
     Calling `show_graph <Composition.show_graph>` with **show_pytorch=True** is sufficient to show the learning
     components used for `Pytorch mode <AutodiffComposition_PyTorch>`.  Using both **show_pytorch** and
     **show_learning** together is redundant, and will issue a warning.  Using **show_learning=True** alone will
     show the standard `learning Components <Composition_Learning_Components>` used for learning in `Python mode
     <AutodiffComposition_Python>`, but may cause an error if the AutodiffComposition has any nested
     AutodiffCompositions (see `note <AutodiffComposition_Show_Learning_with_Nested_Warning>` below).

  .. technical_note::
     An AutodiffComposition's `_build_pytorch_representation <AutodiffComposition._build_pytorch_representation>` method
     can be called to force construction of the `pytorch_representation <AutodiffComposition.pytorch_representation>`
     before the `learn_method


.. _AutodiffComposition_Nesting:

*Nesting*
~~~~~~~~~

An AutodiffComposition can be `nested <Composition_Nested>` inside another Composition for learning, and
there can be any number of such nestings.  However, all of the nested Compositions must be AutodiffCompositions.
As noted `above <AutodiffComposition_PytorchRepresentation>`, the AutodiffComposition is "flattened" when it's
`pytorch_representation <AutodiffComposition.pytorch_representation>` is used for learning in `PyTorch mode
<AutodiffComposition_PyTorch>`; this can be seen by calling the AutodiffComposition's `show_graph
<Composition.show_graph>` method with **show_pytorch=True**.

  .. _AutodiffComposition_Show_Learning_with_Nested_Warning:
  .. warning::
     When `show_graph <Composition.show_graph>` is called for an AutodiffComposition with a nested Composition,
     an error is raised, as standard learning (using `Python mode <AutodiffComposition_Python>` cannot be used;
     instead, use ``show_graph(show_pytorch=True)`` to display the structure of the AutodiffComposition that will
     executed when `PyTorch mode <AutodiffComposition_PyTorch>` is used for learning.

Even though it is flattened, Projections between Nodes at different levels of nesting can still occur if they are
specified for learning. The learning_rate for nested Compositions is inherited from the enclosing Composition unless
it is set individually (see `Composition_Learning_Rate` for a full discussion of how learning rates and precedence of
assignment; see `Composition_Enable_Learning` for enabling and disabling learning in nested Compositions).

  .. technical_note::
     Projections from `Nodes <Composition_Nodes>` in an immediately enclosing outer Composition to the `input_CIM
     <Composition.input_CIM>` of a nested Composition, and from its `output_CIM <Composition.output_CIM>` to Nodes
     in the outer Composition are subject to learning; however those within the nested Composition itself (i.e.,
     from its input_CIM to its INPUT Nodes and from its OUTPUT Nodes to its output_CIM) are *not* subject to learning,
     as they serve simply as conduits of information between the outer Composition and the nested one.

  .. warning::
     Nested Compositions are supported for learning only in `PyTorch mode <AutodiffComposition_PyTorch>`, and
     cause an error if the `learn <Composition.learn>` method of an AutodiffComposition is executed in
     `Python mode <AutodiffComposition_Python>` or `LLVM mode <AutodiffComposition_LLVM>`.


.. _AutodiffComposition_Execution:

Execution
---------

An AutodiffComposition's `run <AutodiffComposition.run>` and `learn <Composition.learn>` methods are the same
as for a `Composition`. However, the **execution_mode** argument has different effects than for a standard Composition.

For `run() <Composition.run>`, execution occurs in `Python mode <AutodiffComposition_Python>`
by default and if either `ExecutionMode.Python` or `ExecutionMode.PyTorch` are specified explicitly
(see `note <AutodiffComposition_PyTorch_Note>` below); `LLVM compilation <AutodiffComposition_LLVM>`
is attempted if one of the  `ExecutionMode.LLVM` modes is specified.

For `learn() <Composition.learn>`, `PyTorch mode <Autodiff_PyTorch>` is used by default, which uses the
`pytorch_representation <AutodiffComposition.pytorch_representation` for execution. Python execution and LLVM
Compilation can be specified explicity (using `ExecutionMode.Python` or `ExecutionMode.LLVMRun`, respectively),
but `restrictions apply. Each mode of exeuction is each described in greater detail below, and summarized in `this
table <Composition_Compilation_Table>`, which provides a comparison of the different modes of execution for an
AutodiffComposition and standard `Composition`.

.. _AutodiffComposition_PyTorch:

*PyTorch mode*
~~~~~~~~~~~~~~

This is the default mode for learning of an AutodiffComposition, but can also be specified explicitly by setting
**execution_mode** = `ExecutionMode.PyTorch` in the `learn() <Composition.learn>` method
(see `example <BasicsAndPrimer_Rumelhart_Model>` in `BasicsAndPrimer`). In this mode, the AutodiffComposition's
`pytorch_representation <AutodiffComposition.pytorch_representation>` is used for learning,
which is about three orders of magntidue faster than `Python mode <AutodiffComposition_Python>`, and
provides additional funtionality (see `above <AutodiffComposition_Additional_Functionality>`). Although
it is best suited for use with `supervised learning <Composition_Learning_Supervised>`, it can also be
used for some forms of `unsupervised learning <Composition_Learning_Unsupervised>` that are supported
in PyTorch (e.g., `self-organized maps <https://github.com/giannisnik/som>`_).

    .. _AutodiffComposition_PyTorch_Note:
    .. note::
       While specifying `ExecutionMode.PyTorch` in the `learn <Composition.learn>` method of an
       AutodiffComposition causes it to use PyTorch for training, specifying this in the `run <Composition.run>`
       method causes it to be executed in `Python mode <AutodiffComposition_Python>` (i.e., using the *Python*
       interpreter, and not PyTorch); this is so that any modulation can take effect during execution, which is
       not supported by PyTorch (see `Control Components <Autodiff_Control_Components>` above).

    .. warning::
      * Specifying `ExecutionMode.LLVMRun` or `ExecutionMode.PyTorch` in the learn() method of a standard
        `Composition` raises an error.

*Execution Sequence*
^^^^^^^^^^^^^^^^^^^^

When PyTorch is used for learning, the AutodiffComposition's `pytorch_representation
<AutodiffComposition.pytorch_representation>` is executed, which is used to implement each `optimization_step
<LearningScale.OPTIMIZATION_STEP>` of the learning process, by calling the relevant forward, backward,
and optimizer_step methods of Pytorch used to implement learning; each optimization_step carries out the following
operations:

  - execute the AutodiffComposition's `forward <AutodiffComposition.autodiff_forward>` method for each stimulus
    in the `minibatch <LearningScale.MINIBATCH>` -- the number of which is specified by the value of `minibatch_size
    <Composition.minibatch_size>` -- to generate the values used to compute the `Losses <Loss>` for each stimulus;

  - aggregate the losses across all stimuli in the minibatch, which is then passed to the AutdoiffCompositon's
    `backward <AutodiffComposition.autodiff_backward>` method to compute the gradients and corresponding weight
    changes for all learnable parameters in the AutodiffComposition;

  - copy the Node values generated in the forward pass and changes to parameters generated in the backward pass and
    optimizer step of the `pytorch_representation <AutodiffComposition.pytorch_representation>`
    to the corresponding Mechanisms' `variables <Mechanism_Base.variable>` and/or `values <Mechanism_Base.value>`,
    and `learnable <MappingProjection.learnable>` Projections' `matrices <MappingProjection.matrix>`) of the
    AutodiffComposition as specified, which can be after each optimizer step, or at the end of the `MINIBATCH` or
    `EPOCH` (see `below <AutodiffComposition_Synchronization_with_PyTorch>`), but always at the end of the `RUN`
    (i.e., call to learn()).


.. _AutodiffComposition_Configuring_Learning:

Which nodes are executed in each optimization step, and which parameters are included in the gradient calculation
can be further customized as described below.

.. _AutodiffComposition_Additional_Optimization_Steps:

*Additional Optimizations Steps*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*optimizatons_per_minibatch*: By default, a single optimization_step is carried out for all of the stimuli in a
`minibatch <LearningScale.MINIBATCH>`. However, as long as there is only one stimulus in a minibatch (i.e.,
`minibatch_size <Composition.minibatch_size>`\\==1), then multiple optimization_steps can be specified for each
stimulus, using the **optimizations_per_minibatch** argument of the AutodiffComposition's constructor (to specify
the default number) or its `learn() <Composition.learn>` method (to specify it for just that execution).
Specifying optimizatons_per_minibatch > 1 can be similar to, but is *not* the same as increasing the `learning_rate
<Composition.learning_rate>` (see `note <Composition_Optimizations_per_Minibatch_Note>`) and, when used with
`execute_in_additional_optimizations <AutodiffComposition.execute_in_additional_optimizations>` can produce important
differences, as described below.

*execute_in_additional_optimizations*: this can be used to specify which `Nodes <Composition_Nodes>` are executed in
which additional `optimization_steps <LearningScale.OPTIMIZATION_STEP>` (i.e., after the first) when more than one
optimization_step is specified. This can be used to implement a form of "online replay" (or `backprop-to-activity
procedure <https://web.stanford.edu/~jlmcc/papers/RogersMcCBook_7_03.pdf>`_) in which a particular part of the model
is given extra optimization_steps to quickly search for a pattern of activity over a subset of Nodes in response to
the stimulus that is useful for some downstream  purpose (see EGO Model for an example). The
**execute_in_additional_optimizations** argument can be specified in either the AutodiffComposition's constructor
(to sepcify a default value) or its `learn() <Compostion.learn>` method (which applies to only that execution).
It is specified as a dict, each key of which is a `Node <Composition_Nodes>` in the AutodiffComposition or one
`nested <AutodiffComposition_Nesting>` within it, and its value is of the following:

  *None* or *True*: execute in all additional optimizations
  COMMENT:
  without any modification(s) to its Parameters;
  COMMENT
  ;

  *False* or *EXCLUDE*: exclude from execution during optimization_steps after the first; this is useful
  primarly when a nested Composition is specified but nodes within it should be excluded
  (e.g., see `note <AutodiffComposition_Nested_Additional_Optimizations>` below);
  COMMENT:
  GIVE EXAMPLE HERE
  COMMENT

  *FIRST*, *LAST*, *ALL* or `range <https://docs.pytorch.org/docs/stable/generated/torch.range.html>`_: include in
  only the first, last, all, or a specified set of additional optimization steps.

  COMMENT:
  BREADCRUMB: IS THIS IMPLEMENTED?
  *(Parameter, value)* or *[(Parameter, value), ...]*: assign specified Parameter values during
    execution of additional optimizations, restoring to previous value(s) for first optimization
    of next trial.
  COMMENT

  .. _AutodiffComposition_Nested_Additional_Optimizations:

  .. note::
     If an AutodiffComposition is specified as a key, then all Nodes within that AutodiffComposition and any nested
     within it are included, except for any explicitly excluded.

COMMENT:
BREADCRUMB: ADD TEXT AND UNCOMM'T ONCE IMPLEMENTED IN CONSTRUCTOR AND LEARN()

.. _AutodiffComposition_Exclusion_From_Gradient_Calculation:

*Exclusion from Gradient Calculation*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`exclude_from_gradient_calc`
COMMENT

.. _AutodiffComposition_Synchronization_with_PyTorch:

*Synchronization of PsyNeuLink Values with PyTorch*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, the outputs (of the modules) and parameters (connection weights) generated in Pytorch during execution
of an AutodiffComposition's `learn() <Composition.learn>` method (using its `pytorch_representation
<AutodiffComposition.pytorch_representation>`) are copied to the corresponding Mechanisms and Projections of the
AutodiffComposition itself at the end of each `run <AutodiffComposition.run>`.  However, this can be cusotmized,
selectively for Mechanism `variables <Mechanism_Base.variable>` or `values <Mechanism_Base.value>`, Projection
`matrices<MappingProjection.matrix>`, and/or the Composition `results <Composition.results>`, to occur after each
`optimization_step <LearningScale.OPTIMIZATION_STEP>`, `minibatch <LearningScale.MINIBATCH>`, `trial
<LearningScale.TRIAL>`, training `epoch <LearningScale.EPOCH>`, full `run <LearningScale.RUN>`, or not at all.
This can be specified using following arguments of either the AutodiffComposition's constructor or `learn()
<Composition.learn>` method:

    - synch_projection_matrices_with_torch : `OPTIMIZATION_STEP`, `MINIBATCH`, `EPOCH` or `RUN`
    - synch_node_variables_with_torch : `OPTIMIZATION_STEP`, `TRIAL`, `MINIBATCH`, `EPOCH`, `RUN` or None
    - synch_node_values_with_torch : `OPTIMIZATION_STEP`, `MINIBATCH`, `EPOCH` or `RUN`
    - synch_results_with_torch : `OPTIMIZATION_STEP`, `MINIBATCH`, `EPOCH` or `RUN`

    .. note::
        Copying more frequently keeps the PsyNeuLink components more closely synchronized with the corresponding
        Pytorch elements of the `pytorch_representation <AutodiffComposition.pytorch_representation>` during learning,
        which can be useful for debugging and/or monitoring the learning process in Pytorch; but can slow performance.

.. _AutodiffComposition_Save_Pytorch_Training_Data:

*Saving Pytorch Training Data*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the `samples <AutodiffComposition_Sample>`, `targets <AutodiffComposition_Target>`, and `losses
<AutodiffComposition_LossMechanism>` are stored for the last stimulus of each `MINIBATCH`. However, this can be
customized to occur for each `OPTIMIZATION_STEP`, `EPOCH`, `RUN`, or not at all (using ``None``) by specifying one
of these values for the following Parameters, using the corresponding argument in either the AutodiffComposition's
constructor (to specify the default vaue) or its `learn() <Composition.learn>` method (to specify the value used for
that execution):
`retain_torch_sample_values <AutodiffComposition.retain_torch_sample_values>`,
`retain_torch_targets <AutodiffComposition.retain_torch_targets>`,
or `retain_torch_losses <AutodiffComposition.retain_torch_losses>`.

.. _AutodiffComposition_Python:

*Python mode*
~~~~~~~~~~~~~
An AutodiffComposition can also be run using the standard PsyNeuLink learning components.  However, this cannot
be used if the AutodiffComposition has any nested Compositions, irrespective of whether they are ordinary
Compositions or AutodiffCompositions; nor can it be used to specify internal *targets*.

.. _AutodiffComposition_LLVM:

*LLVM mode*
~~~~~~~~~~~
This is specified by setting **execution_mode** = `ExecutionMode.LLVMRun` in the `learn <Composition.learn>`
method of an AutodiffCompositon. This provides the fastest performance, but is limited to `supervised learning
<Composition_Learning_Supervised>` using the `BackPropagation` algorithm, and does not support learning of `nested
 Compositions <Composition_Nested>` nor subclasses of AutodiffComposition that rely on PyTorch (e.g.,
 `GRUComposition` and `EMComposition`) -- `PyTorch mode <AutodiffComposition_PyTorch>` should be used for these.
LLVMRun can be used with standard forms of loss, including mean squared error (MSE) and cross entropy, by specifying
this in the **loss_spec** argument of the constructor (see `AutodiffComposition <AutodiffComposition_Class_Reference>`
for additional details, and `Compilation Modes <Composition_Compiled_Modes>` for more information about executing a
Composition in compiled mode.

    .. note::
       Specifying `ExecutionMode.LLVMRun` in either the `learn <Composition.learn>` and `run <Composition.run>`
       methods of an AutodiffComposition causes it to (attempt to) use compiled execution in both cases; this is
       because LLVM compilation supports the use of modulation in PsyNeuLink models (as compared to `PyTorch mode
       <AutodiffComposition_PyTorch>`; see `note <AutodiffComposition_PyTorch_Note>` below).

COMMENT:
.. _AutodiffComposition_Nested_Modulation:

*Nested Execution and Modulation*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like any other `Composition`, an AutodiffComposition can be `nested <Composition_Nested>` inside another
(see `example <AutodiffComposition_Nested_Example>` below). However, during learning, none of the internal
Components of the AutodiffComposition (e.g., intermediate layers of a neural network model) are accessible to the
other Components of the outer Composition, (e.g., as sources of information, or for `modulation
<ModulatorySignal_Modulation>`).  However, when it is executed using its `run <Composition.run>` method, an
AutodiffComposition functions like any other, and all of its internal Components are accessible to other Components
of the outer Composition. Thus, as long as access to its internal Components is not needed during learning, an
`AutodiffComposition` can be trained, and then used to execute the trained Composition like any other.
COMMENT

.. _AutodiffComposition_Logging:

*Logging*
~~~~~~~~~

Logging in AutodiffCompositions follows the same procedure as `logging in a Composition <Log>`. However, since an
AutodiffComposition internally converts all of its Mechanisms either to an equivalent PyTorch module (or to LLVM in
`LLVM mode <AutodiffComposition_LLVM>`), then its inner components are not actually executed. This means that there
is limited support for logging parameters of components inside an AutodiffComposition; Currently, the only supported
parameters are the:

- `matrix <MappingProjection.matrix>` parameter of `MappingProjection <MappingProjection>`;

- `value <Mechanism_Base.value>` parameter of its `Mechanisms <Mechanism>`.


.. _AutodiffComposition_Examples:

Examples
--------

.. _AutodiffComposition_Creation_Example:

The following is an example showing how to create a simple AutodiffComposition, specify its inputs and targets,
and run it with learning enabled and disabled:

    >>> import psyneulink as pnl
    >>> # Set up PsyNeuLink Components
    >>> my_mech_1 = pnl.TransferMechanism(function=pnl.Linear, input_shapes = 3)
    >>> my_mech_2 = pnl.TransferMechanism(function=pnl.Linear, input_shapes = 2)
    >>> my_projection = pnl.MappingProjection(matrix=np.random.randn(3,2),
    ...                     sender=my_mech_1,
    ...                     receiver=my_mech_2)
    >>> # Create AutodiffComposition
    >>> my_autodiff = pnl.AutodiffComposition()
    >>> my_autodiff.add_node(my_mech_1)
    >>> my_autodiff.add_node(my_mech_2)
    >>> my_autodiff.add_projection(sender=my_mech_1, projection=my_projection, receiver=my_mech_2)
    >>> # Specify inputs and targets
    >>> my_inputs = {my_mech_1: [[1, 2, 3]]}
    >>> my_targets = {my_mech_2: [[4, 5]]}
    >>> input_dict = {"inputs": my_inputs, "targets": my_targets, "epochs": 2}
    >>> # Run Composition in learnng mode
    >>> my_autodiff.learn(inputs = input_dict)
    >>> # Run Composition in test mode
    >>> my_autodiff.run(inputs = input_dict['inputs'])


.. _AutodiffComposition_Nested_Example:

The following shows how the AutodiffComposition created in the previous example can be nested and run inside another
Composition::

    >>> # Create outer composition
    >>> my_outer_composition = pnl.Composition()
    >>> my_outer_composition.add_node(my_autodiff)
    >>> # Specify dict containing inputs and targets for nested Composition
    >>> training_input = {my_autodiff: input_dict}
    >>> # Run in learning mode
    >>> result1 = my_outer_composition.learn(inputs=training_input)
    COMMENT:
    >>> # Run with learning disabled (and standard input format)
    >>> no_training_input = {my_autodiff: my_inputs}
    >>> result2 = parentmy_outer_compositionComposition.run(inputs=no_training_input)
    COMMENT

.. _AutodiffComposition_Class_Reference:

Class Reference
---------------

"""
import copy
import logging
import os
import warnings
import numpy as np
from packaging import version
from pathlib import Path, PosixPath
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Set, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.optim as optim
    torch_available = True
except ImportError:
    torch_available = False
else:
    from psyneulink.library.compositions.pytorchshowgraph import PytorchShowGraph

from psyneulink._typing import Iterable, Mapping, Optional, Literal
from psyneulink.core.components.component import Component
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import (
    ProcessingMechanism, ProcessingMechanism_Base)
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import LearningMechanism
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection, PROXY_FOR
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.compositions.composition import (
    Composition,
    CompositionError,
    LearningScale,
    OptimizerParams,
    SampleTargetPair,
    SampleTargetSpec,
)
from psyneulink.core.compositions.noderoles import NodeRole
from psyneulink.core.compositions.report import (ReportOutput, ReportParams, ReportProgress, ReportSimulations,
                                                 ReportDevices, EXECUTE_REPORT, LEARN_REPORT, PROGRESS_REPORT)
from psyneulink.library.components.mechanisms.processing.objective.lossmechanism import LossMechanism
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import (
    ALL,
    AUTODIFF_COMPOSITION,
    DEFAULT,
    DEFAULT_LEARNING_RATE,
    ERROR,
    EXECUTION_MODE,
    LEARNING_RATE,
    Loss,
    LOSSES,
    MATRIX_WEIGHTS,
    NAME,
    NODE_VALUES,
    NODE_VARIABLES,
    PROJECTION,
    RESULTS,
    RETAIN_IN_PNL_OPTIONS,
    SAMPLE,
    SINGLETON,
    SOFT_CLAMP,
    SYNCH_WITH_PNL_OPTIONS,
    TARGET,
    TARGETS,
    SAMPLE_VALUES,
    WARNING
)
from psyneulink.core.globals.utilities import (is_identity_matrix, is_matrix_keyword, is_numeric,
                                               convert_to_list, deprecation_warning, counts)
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core import llvm as pnlvm


if TYPE_CHECKING:
    from psyneulink.library.compositions.grucomposition.pytorchGRUwrappers import DummyProjection


logger = logging.getLogger(__name__)


__all__ = [
    'AutodiffComposition', 'OPTIMIZER_PARAMS', 'EXCLUDE_FROM_GRADIENT_CALC',
]

OPTIMIZER_PARAMS = 'optimizer_params'
EXCLUDE_FROM_GRADIENT_CALC = 'exclude_from_gradient_calc'
CONSTRUCTOR_TARGETS = "AutodiffComposition constructor 'targets'"

SynchRetainArg = Optional[Union[LearningScale, str]]


class AutodiffCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AutodiffComposition(Composition):
    """
    AutodiffComposition(                           \
        optimizer_type='sgd',                      \
        loss_spec=Loss.MSE,                        \
        targets=None,                              \
        weight_decay=0,                            \
        enable_learning=True,                      \
        learning_rate=0.001,                       \
        execute_in_additional_optimizations=None   \
        synch_projection_matrices_with_torch=RUN,  \
        synch_node_variables_with_torch=None,      \
        synch_node_values_with_torch=RUN,          \
        synch_results_with_torch=RUN,              \
        retain_torch_sample_values=MINIBATCH,      \
        retain_torch_targets=MINIBATCH,            \
        retain_torch_losses=MINIBATCH,             \
        device=CPU
        )

    Subclass of `Composition` that trains models using either LLVM compilation or `PyTorch <https://pytorch.org>`_;
    see and `Composition <Composition_Class_Reference>` for additional arguments and attributes.  See `Composition`
    for additional arguments to constructor.

    Arguments
    ---------

    optimizer_type : str : default 'sgd'
        the kind of optimizer used in training. The current options are 'sgd' or 'adam'.

    loss_spec : Loss or PyTorch loss function : default Loss.MSE
        specifies the default loss function for training; see `Loss` for arguments;
        any specifications in **targets** override this default.

    targets : LossMechanism, tuple, list or dict  : default None
        specifies the target(s) used for training the model;
        see `AutodiffComposition_Target_Specification` for additional details;

    weight_decay : float : default 0
        specifies the L2 penalty (which discourages large weights) used by the optimizer.

    enable_learning : bool: default True
        specifies whether the AutodiffComposition should enable learning when run in `learning mode
        <Composition.learn>` (see `Composition_Enable_Learning` for additional details).

    learning_rate : float, int, bool or dict : default 0.001
        specifies the learning rate(s) passed to the optimizer; overridden by any specified in the `learn
        <AutdodiffComposition.learn>` method of the AutodiffComposition; if a dict is used, and it does
        not contain an entry for *DEFAULT_LEARNING_RATE*, the default indicated above is used (see `learning_rate
        (see `AutodiffComposition_Learning_Rate` and `Composition_Learning_Rate` for additional details).

    execute_in_additional_optimizations : dict{Node: [<bool | *EXCLUDE* | (Parameter, value)]} (default None)
        specifies which `Nodes <Composition_Nodes>` of the AutodiffComposition should be included in the forward pass
        for any additional optimization steps after the first (see `AutodiffComposition_Optimization_Steps`
        for fuller explanation and additional details of specification).

    synch_projection_matrices_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy Pytorch parameters to PsyNeuLink
        `Projection matrices <MappingProjection.matrix>` (connection weights), which can be overridden by specifying
        the **synch_projection_matrices_with_torch** argument in the `learn <Composition.learn>` method (see
        `LearningScale` for information about settings, an `AutodiffComposition_Synchronization_with_PyTorch` for
        additional details).

    synch_node_variables_with_torch : `LearningScale` : default None
        specifies the default for the AutodiffComposition for when to copy the current input to Pytorch nodes to
        the PsyNeuLink `variable <Mechanism_Base.value>` of the corresponding PsyNeuLink `Nodes <Composition_Nodes>`,
        which can be overridden by specifying the **synch_node_variables_with_torch** argument in the `learn
        <Composition.learn>` method (see `LearningScale` for information about settings, and
        `AutodiffComposition_Synchronization_with_PyTorch` for additional details).

    synch_node_values_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy the current output of Pytorch nodes to the
        PsyNeuLink `value <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `nodes <Composition_Nodes>`,
        which can be overridden by specifying the **synch_node_values_with_torch** argument in the `learn
        <Composition.learn>` method (see `LearningScale` for information about settings, and
        `AutodiffComposition_Synchronization_with_PyTorch` for additional details).

    synch_results_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy the outputs of the Pytorch model
        to the AutodiffComposition's `results <Composition.results>` attribute, which can be overridden by
        specifying the **synch_results_with_torch** argument in the `learn <Composition.learn>` method.
        Note that this differs from **retain_torch_sample_values**, which specifies the frequency at which
        the outputs of the PyTorch model are tracked, all of which are stored in the AutodiffComposition's
        `torch_sample_values <AutodiffComposition.torch_sample_values>` attribute at the end of the run
        (see `LearningScale` for information about settings, an
        `AutodiffComposition_Synchronization_with_PyTorch` for additional details).

    retain_torch_sample_values : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for the scale at which the outputs of the Pytorch
        model are tracked, all of which are stored in the AutodiffComposition's `torch_sample_values
        <AutodiffComposition.torch_sample_values>` attribute at the end of the run; this can be overridden
        by specifying the **retain_torch_sample_values** argument in the `learn <Composition.learn>` method.
        Note that this differs from **synch_results_with_torch**, which specifies the frequency with
        which values are copied to the AutodiffComposition's `results` attribute (see `retain_torch_sample_values
        <AutodiffComposition.retain_torch_sample_values>` for additional details).

    retain_torch_targets : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for when to copy the targets used for training the Pytorch
        model to the AutodiffComposition's `torch_targets <Composition.torch_targets>` attribute, which can be
        overridden by specifying the **retain_torch_targets** argument in the `learn <Composition.learn>`
        method (see `retain_torch_targets <AutodiffComposition.retain_torch_targets>` for additional details).

    retain_torch_losses : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for the scale at which the losses of the Pytorch model
        are tracked, all of which are stored in the AutodiffComposition's `torch_losses <Composition.torch_losses>`
        attribute at the end of the run (see `retain_torch_losses <AutodiffComposition.retain_torch_losses>` for
        additional details).

    device : torch.device : default device-dependent
        specifies the device on which the model is run. If None, the device is set to 'cuda' if available,
        then 'mps`, otherwise 'cpu'.

    Attributes
    ----------

    pytorch_representation : PytorchCompositionWrapper
        represents the PyTorch model of the AutodiffComposition, which is created when the AutodiffComposition is
        run in `PyTorch mode <AutodiffComposition_PyTorch>`.

    optimizer : PyTorch optimizer function
        the optimizer used for training. Depends on the **optimizer_type**, **learning_rate**, and **weight_decay**
        arguments from initialization.

    loss_spec : PyTorch loss function
        the loss function used for training. Depends on the **loss_spec** argument from initialization.

    targets : list of LossMechanisms
        each LossMechanism computes the loss for the output of the Node from which it recieves its *SAMPLE* input
        (the "student" Node) by comparing it to the output of the Node from which it receives its *TARGET* input
        (the "teacher" Node), using the specified loss function; see `AutodiffComposition_Target` for additional
        details.

    learning_rate : float or bool
        determines the default learning_rate passed the `optimizer <PytorchCompositionWrappe.optimizer>`,
        that is applied to all `Projections <Projection>` in the AutodiffComposition that are `learnable
        <MappingProjection.learnable>`, and for which individual rates have not been specified (see
        `AutodiffComposition_Learning_Rates` for additional details).

    execute_in_additional_optimizations : dict{Node:[(Parameter, value)]}
        determines which `Nodes <Composition_Nodes>` of the AutodiffComposition should be included in the forward
        pass for any additional optimization steps after the first (see `AutodiffComposition_Optimization_Steps`
        for additional information).

    synch_projection_matrices_with_torch : OPTIMIZATION_STEP, MINIBATCH, EPOCH or RUN
        determines when to copy PyTorch parameters to PsyNeuLink `Projection matrices <MappingProjection.matrix>`
        (connection weights) if this is not specified in the call to `learn <Composition.learn>` (see
        `AutodiffComposition_Synchronization_with_PyTorch` for additional details).

    synch_node_variables_with_torch : OPTIMIZATION_STEP, TRIAL, MINIBATCH, EPOCH, RUN or None
        determines when to copy the current input to Pytorch functions to the PsyNeuLink `variable
        <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `Nodes <Composition_Nodes>`,
        if this is not specified in the call to `learn <Composition.learn>`
        COMMENT:
        BREADCRUMB: 3/15/25 ADD EXPLANATION OF WHY THIS IS NOT GENERALLY USEFUL ALONG THE LINES OF THE FOLLOWING
                 ALSO RELATE TO EXECUTE_NODES OPTION ONCE IMPLEMENTED
        . This is supported for inspection and debugging, but is not generally useful, as PsyNeuLink uses `Lazy
        Evaluation <Component_Lazy_Updating>`, in which the variable of a node is determined by the input it receives
        during execution.
        COMMENT
        (see `AutodiffComposition_Synchronization_with_PyTorch` for additional details)

    synch_node_values_with_torch : OPTIMIZATION_STEP, MINIBATCH, EPOCH or RUN
        determines when to copy the current output of Pytorch functions to the PsyNeuLink `value
        <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `Nodes <Composition_Nodes>`,
        if this is not specified in the call to `learn <Composition.learn>` (see
        `AutodiffComposition_Synchronization_with_PyTorch` for additional details).

    synch_results_with_torch : OPTIMIZATION_STEP, TRIAL, MINIBATCH, EPOCH or RUN
        determines when to copy the current outputs of Pytorch nodes to the PsyNeuLink `results <Composition.results>`
        attribute of an AutodiffComposition if this is not specified in the call to `learn <Composition.learn>`
        (see `AutodiffComposition_Synchronization_with_PyTorch` for additional details).

    retain_torch_sample_values : OPTIMIZATION_STEP, MINIBATCH, EPOCH, RUN or None
        determines the scale at which the outputs of the Pytorch model are tracked, all of which are stored in the
        AutodiffComposition's `results <Composition.results>` attribute at the end of the run if this is not specified
        in the call to `learn <Composition.learn>` (see `LearningScale` for information about settings).

    retain_torch_targets : OPTIMIZATION_STEP, TRIAL, MINIBATCH, EPOCH, RUN or None
        determines the scale at which the targets used for training the Pytorch model are tracked, all of which
        are stored in the AutodiffComposition's `targets <AutodiffComposition_Target>` attribute at the end of the run
        if this is not specified in the call to `learn <Composition.learn>`
        (see `LearningScale` for information about settings).

    retain_torch_losses : OPTIMIZATION_STEP, MINIBATCH, EPOCH, RUN or None
        determines the scale at which the losses of the Pytorch model are tracked, all of which are stored in
        the AutodiffComposition's `torch_losses <AutodiffComposition.torch_losses>` attribute at the end of the run
        if this is nota specified in the call to `learn <Composition.learn>`
        (see `LearningScale` for information about settings).

    torch_parameters : List[Tuple[str, torch.nn.parameter]]
        list of PyTorch named_parameters() for `pytorch_representation <AutodiffComposition.pytorch_representation>`
        of AutodiffComposition.

    torch_sample_values : List[ndarray]
        stores the outputs (converted to np arrays) of the Pytorch model trained during learning, at the frequency
        specified by `retain_torch_sample_values <AutodiffComposition.retain_torch_sample_values>` if it is set
        to *MINIBATCH*, *EPOCH*, or *RUN*; see `retain_torch_sample_values
        <AutodiffComposition.retain_torch_sample_values>` for additional details.

    torch_targets : List[ndarray]
        stores the targets used for training the Pytorch model during learning at the frequency specified by
        `retain_torch_targets <AutodiffComposition.retain_torch_targets>` if it is set to *MINIBATCH*, *EPOCH*,
        or *RUN*; see `retain_torch_targets <AutodiffComposition.retain_torch_targets>` for additional details.

    torch_losses : list of floats
        stores the average loss after each weight update (i.e. each minibatch) during learning, at the frequency
        specified by `retain_torch_sample_values <AutodiffComposition.retain_torch_sample_values>` if it is set to *MINIBATCH*,
        *EPOCH*, or *RUN*; see `retain_torch_losses <AutodiffComposition.retain_torch_losses>` for additonal details.

    COMMENT:  FIX: NOT CURRENTLY BEING POPULTED, BUT SEEMS TO BE USED BY _get_total_loss() and early_stopper
    trial_losses = Parameter([])
    COMMENT

    last_saved_weights : path
        path for file to which weights were last saved.

    last_loaded_weights : path
        path for file from which weights were last loaded.

    device : torch.device
        the device on which the model is run.

    full_sequence_mode: bool : default False
        Whether to run the underlying Composition in full sequence mode or not. In full sequence mode, each element of
        an input sequence for a trial is processed in a separate time step. This is needed only if there are sequential
        dependencies between the mechanisms of the compositions. Note, if the composition contains GRU compositions
        wrappers full sequence mode is not needed (and should be avoided to improve efficiency) because the composition
        wrapper itself handles the sequential dependencies between the mechanisms of the GRU composition.

    """

    componentCategory = AUTODIFF_COMPOSITION
    if torch_available:
        from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper
        pytorch_composition_wrapper_type = PytorchCompositionWrapper
        pytorch_mechanism_wrapper_type = PytorchMechanismWrapper

    class Parameters(Composition.Parameters):
        pytorch_representation = None
        # optimizer = None
        loss_spec = Parameter(Loss.MSE, stateful=False, modulable=False)
        targets = Parameter(None, fallback_value=None, structural=True, stateful=False, modulable=False,
                            dependencies={'loss_spec'})
        synch_projection_matrices_with_torch = Parameter(LearningScale.RUN)
        synch_node_variables_with_torch = Parameter(None)
        synch_node_values_with_torch = Parameter(LearningScale.RUN)
        synch_results_with_torch = Parameter(LearningScale.RUN)
        retain_torch_sample_values = Parameter(LearningScale.MINIBATCH)
        retain_torch_targets = Parameter(LearningScale.MINIBATCH)
        retain_torch_losses = Parameter(LearningScale.MINIBATCH)
        trial_losses = Parameter([]) # FIX <- related to early_stopper, but not getting assigned anywhere
        device = None

        def _validate_loss_spec(self, spec):
            if spec and spec not in Loss and torch_available and not isinstance(torch.nn.modules.loss._Loss):
                return f"must be a member of the Loss enum or a PyTorch loss function."

        def _parse_targets(self, specs)->list:
            """Parse targets argument to standardize into list of LossMechanisms or (sample, target) tuples
            Convert Mechanism specs for sample and/or target in a tuple to the corresponding primary port.
            """
            if isinstance(specs, (LossMechanism, tuple, set, dict, list)):
                specs = convert_to_list(specs)
            return specs

        def _validate_targets(self, spec):
            if spec is None:
                return None
            if not isinstance(spec, list):
                return (f"should be a dict of sample:target pairs or, alternatively, "
                        f"a set or list of (sample, target) tuples.")
            for item in spec:
                if not isinstance(item, (LossMechanism, tuple)):
                    return (f"must be a (sample, target) tuple, LossMechanism, or a list containing these.")
                if isinstance(item, tuple):
                    if not isinstance(item[0], (OutputPort, ProcessingMechanism_Base)):
                        return (f"sample specification must be an OutputPort or ProcessingMechanism (got: {item[0]}).")
                    if isinstance(item[0], OutputPort) and not isinstance(item[0].owner, ProcessingMechanism_Base):
                        return (f"sample specification must be a ProcessingMechanism or the OutputPort of one (got: {item[0]}).")
                    if not isinstance(item[1], (OutputPort, ProcessingMechanism_Base, str)):
                        return (f"target specificadtion must be an OutputPort, ProcessingMechanism "
                                f"or the keyword '{TARGET}' (got: {item[1]}).")
                    if isinstance(item[1], OutputPort) and not isinstance(item[1].owner, ProcessingMechanism_Base):
                        return (f"target specification must be a ProcessingMechanism or the OutputPort of one "
                                f"(got: {item[1]}).")
                    if isinstance(item[1], str) and item[1] != TARGET:
                        return (f"the only keyword that can be used for the target specification is '{TARGET}' "
                                f"(got: {item[1]}).")
                    # assert isinstance(item[0], OutputPort), \
                    #     ("PROGRAM ERROR: 1st item of tuple specification for targets arg should be OutputPort by now.")
                    # assert isinstance(item[1], OutputPort) or item[1] == TARGET, \
                    #     ("PROGRAM ERROR: 2nd item of tuple specification for targets arg should be OutputPort by now.")

        def _parse_LearningScale_param(self, value):
            try:
                return LearningScale(value)
            except ValueError:
                return value

        # TODO: consider implementing a type/enum parser as attr on Parameter
        _parse_synch_projection_matrices_with_torch = _parse_LearningScale_param
        _parse_synch_node_variables_with_torch = _parse_LearningScale_param
        _parse_synch_node_values_with_torch = _parse_LearningScale_param
        _parse_synch_results_with_torch = _parse_LearningScale_param
        _parse_retain_torch_sample_values = _parse_LearningScale_param
        _parse_retain_torch_targets = _parse_LearningScale_param
        _parse_retain_torch_losses = _parse_LearningScale_param

        def _validate_LearningScale_param(
            self, value: Any, invalid_members: Optional[Set] = None
        ):
            # NOTE: this occurs after parse, so it will already be a
            # LearningScale if possible
            if value is None:
                return None

            try:
                is_learningscale = value in LearningScale
            except TypeError:
                is_learningscale = False

            try:
                is_invalid_member = value in invalid_members
            except TypeError:
                is_invalid_member = False

            if not is_learningscale:
                return (
                    f"must be a {LearningScale.__name__} or corresponding"
                    f" string: {', '.join(map(str, LearningScale))}"
                )

            if is_invalid_member:
                valid = set(LearningScale).difference(invalid_members)
                return f"can't be used; use another value of: {', '.join(valid)}"

            return None

        def _validate_synch_projection_matrices_with_torch(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_synch_node_variables_with_torch(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_synch_node_values_with_torch(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_synch_results_with_torch(self, spec):
            return self._validate_LearningScale_param(spec, {LearningScale.OPTIMIZATION_STEP})

        def _validate_retain_torch_sample_values(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_retain_torch_targets(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_retain_torch_losses(self, spec):
            return self._validate_LearningScale_param(spec)

    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    @check_user_specified
    def __init__(self,
                 pathways=None,
                 optimizer_type: str = 'sgd',
                 loss_spec: Loss = Loss.MSE, # default is Loss.MSE set in Parameters
                 targets: Union[LossMechanism, tuple, list, dict] = None,
                 weight_decay: float = 0.0,
                 learning_rate: Optional[Union[float,int,bool,dict,]]=.001,
                 enable_learning: bool = True,
                 execute_in_additional_optimizations=None,
                 force_no_retain_graph: bool = False,
                 refresh_losses: bool = False,
                 synch_projection_matrices_with_torch: SynchRetainArg = LearningScale.RUN,
                 synch_node_variables_with_torch: SynchRetainArg = None,
                 synch_node_values_with_torch: SynchRetainArg = LearningScale.RUN,
                 synch_results_with_torch: SynchRetainArg = LearningScale.RUN,
                 retain_torch_sample_values: SynchRetainArg = LearningScale.MINIBATCH,
                 retain_torch_targets: SynchRetainArg = LearningScale.MINIBATCH,
                 retain_torch_losses: SynchRetainArg = LearningScale.MINIBATCH,
                 device=None,
                 disable_cuda=True,
                 cuda_index=None,
                 full_sequence_mode: bool = False,
                 name="autodiff_composition",
                 **kwargs):

        show_graph_attributes = kwargs.pop('show_graph_attributes', {})

        # Deal with deprecated arg
        if OPTIMIZER_PARAMS in kwargs:
            opt_params_arg = deprecation_warning(self, kwargs,
                                                 deprecated_args={OPTIMIZER_PARAMS:LEARNING_RATE},
                                                 additional_msg=" Other torch.nn.optimizer parameters are not "
                                                                "currently supported, but will be in a future version.")
            if learning_rate is not None:
                opt_params_arg[DEFAULT_LEARNING_RATE] = learning_rate
            learning_rate = opt_params_arg.pop(LEARNING_RATE)


        super(AutodiffComposition, self).__init__(
            name = name,
            pathways=pathways,
            optimizer_type = optimizer_type,
            loss_spec = loss_spec,
            targets = targets,
            weight_decay = weight_decay,
            enable_learning = enable_learning,
            learning_rate = learning_rate,
            synch_projection_matrices_with_torch = synch_projection_matrices_with_torch,
            synch_node_variables_with_torch = synch_node_variables_with_torch,
            synch_node_values_with_torch = synch_node_values_with_torch,
            synch_results_with_torch = synch_results_with_torch,
            retain_torch_sample_values = retain_torch_sample_values,
            retain_torch_targets = retain_torch_targets,
            retain_torch_losses = retain_torch_losses,
            **kwargs)

        self._built_pathways = False
        self.loss_mechs_map = {}  # {LossMechanism : (sample, target)} tuple of sender Ports
        self._trained_comp_nodes_to_pytorch_nodes_map = None # Set by subclasses that replace trained OUTPUT Nodes
        self._input_comp_nodes_to_pytorch_nodes_map = None # Set by subclasses that replace INPUT Nodes
        self._pytorch_projections = []
        self._constructor_target_specs = {}
        self.optimizer_type = optimizer_type
        self.force_no_retain_graph = force_no_retain_graph
        self.refresh_losses = refresh_losses
        self.weight_decay = weight_decay
        self.loss_function = None
        self.last_saved_weights = None
        self.last_loaded_weights = None
        self.full_sequence_mode = full_sequence_mode
        self.execute_in_additional_optimizations = execute_in_additional_optimizations or {}

        # keeps track of average loss per epoch
        self.losses = []

        # ordered execution sets for the pytorch model
        self.execution_sets = None

        if not disable_cuda and torch.cuda.is_available():
            if cuda_index is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:' + str(cuda_index))
        elif torch_available:
            self.device = torch.device('cpu')
            self.torch_dtype = self.pytorch_composition_wrapper_type.torch_dtype
        else:
            self.device = device
            self.torch_dtype = None

        # Avoid repeated warnings; flagss are set to True after first warning about:
        # - failure to specify execution mode
        self._warned_about_default_execution_mode = False
        # - specyfing a target as TARGET in constructor but no value provided in learn()
        self._warned_about_unspecified_target_in_learn = False
        # - using show_learning with show_pytorch
        self._warned_about_unecessary_show_learning_arg_in_call_to_show_graph = False
        # - modulatory components that will not execute in run() or learn() in ExecutionMode.PyTorch
        self._warned_about_modulatory_components = False
        # - target appears before sample in the same pathway
        self._warned_about_target_before_sample_in_pathway = False
        # - no learnable Projections in the Composition
        self._warned_about_no_learnable_projections = False
        # torch params added when warned in copy_projection_matrix_to_torch_param() to avoid repeats for same param
        self.require_grad_warning = []

        # ShowGraph
        self.assign_ShowGraph(show_graph_attributes)

    def assign_ShowGraph(self, show_graph_attributes):
        """Override to replace assignment of ShowGraph class with PytorchShowGraph if torch is available"""
        show_graph_attributes = show_graph_attributes or {}
        if torch_available:
            self._show_graph = PytorchShowGraph(self, **show_graph_attributes)
        else:
            from psyneulink.core.compositions.showgraph import ShowGraph
            self._show_graph = ShowGraph(self, **show_graph_attributes)

    @handle_external_context()
    def infer_backpropagation_learning_pathways(self, execution_mode, context=None, base_context=None)->list:
        """Create backpropagation learning pathways for every INPUT Node --> OUTPUT Node pathway
        Pathways are constructed in _get_pytorch_backprop_pathways()
            Flattens nested compositions:
              - only includes the Projections in outer Composition to/from the CIMs of the nested Composition
                (i.e., to input_CIMs and from output_CIMs) -- the ones that should be learned;
              - excludes Projections from/to CIMs in the nested Composition
                (from input_CIMs and to output_CIMs), as those should remain identity Projections;
              see `PytorchCompositionWrapper` for table of how Projections are handled and further details.
        For Python mode:
          - calls add_backpropagation_learning_pathway() for each identified pathway
            which also creates TARGET Nodes for TERMINAL Nodes in each pathway
        For PyTorch mode:
          - if **targets** are specified in the AutodiffComposition constructor,
             LossMechanisms and MappingProjections are constructed for them;
          - otherwise, TERMINAL Nodes of each pathway are used to construct LossMechanisms and TARGET Nodes
            with associated MappingProjections) to allow targets to be specified in inputs argument of learn().
          - the above allow:
            - trial-by-trial losses to be kept aligned with inputs in batch / minibatch construction
            - losses to be tracked for logging (as mechs of a Composition)
        Return list of LossMechanisms and TARGET Nodes
        """
        context = context or Context()
        base_context = base_context or Context()

        # Construct a pathway(s) for each INPUT Node (including BIAS Nodes), except the TARGET Node)
        self.pytorch_backprop_pathways = self._get_pytorch_backprop_pathways(context)

        if execution_mode is pnlvm.ExecutionMode.PyTorch:
            # Construct LossMechanisms, and TARGET Nodes if needed, for inclusion in pathway construction below
            self._instantiate_loss_components(self.pytorch_backprop_pathways, context, base_context)

        else:
        # if execution_mode is not pnlvm.ExecutionMode.PyTorch:
            # For non-Pytorch modes, construct and add PNL backpropagation learning pathways for each INPUT Node
            #    that will construct learning components, including TARGET Nodes for all TERMINAL Nodes
            for pathway in self.pytorch_backprop_pathways:
                self.add_backpropagation_learning_pathway(pathway=pathway,
                                                          loss_spec=self.loss_spec)

        self._analyze_graph()
        return self.learning_components

    @handle_external_context()
    def _get_pytorch_backprop_pathways(self, context)->list:
        """Get backpropagation pathways for all INPUT Nodes of AutodiffComposition
        Return a list of all pathways
        """
        self._analyze_graph()
        return [pathway
                    for node in (self.get_nodes_by_role(NodeRole.INPUT) + self.get_nodes_by_role(NodeRole.BIAS))
                    if node not in self.get_nodes_by_role(NodeRole.TARGET)
                    for pathway in self._get_pytorch_backprop_pathway(node, context)]

    def _get_pytorch_backprop_pathway(self, input_node, context)->list:
        """Breadth-first search from input_node to find all input -> <any OUTPUT Node> pathways
        Uses queue((node, afferent Projection, composition) to traverse all nodes in the graph
        IMPLEMENTATION NOTE:  flattens nested Compositions, removing any CIMs in the nested Compositions
        Return a list of all pathways from input_node -> any OUTPUT Node
        """

        pathways = []  # List of all feedforward pathways from INPUT Node to OUTPUT Node
        dependency_dict = {}      # Dictionary of previous component for each component in every pathway
        afferent = input_node.input_port.path_afferents[0] if input_node.input_port.path_afferents else None
        queue = deque([(input_node, afferent, self)])  # Queue of nodes to visit in breadth-first

        def create_pathway(current_comp, terminal_node:ProcessingMechanism_Base, afferent_proj:MappingProjection)->list:
            """Create pathway starting with node (presumably an output NODE) and working backward via dependency_dict"""
            pathway = []
            if isinstance(afferent_proj.receiver.owner, CompositionInterfaceMechanism):
                cim_output_port = afferent_proj.receiver.owner._get_output_port_for_input_port(afferent_proj.receiver)
                afferent_proj = cim_output_port.efferents[0]
            assert terminal_node == afferent_proj.receiver.owner, \
                (f"PROGRAM ERROR: Bad afferent_proj passed to _create_pathways "
                 f"for OUTPUT Node {terminal_node.name} in {self.name}")
            entry = (afferent_proj.receiver, afferent_proj.receiver.path_afferents.index(afferent_proj))
            # TEACHER_TARGET BREADCRUMB: entry in dependency_dict FAILS TO MAKE IT PAST input_CIM
            #                            SINCE CIMs ARE NOT IN dependency_dict, AS LEAST NOT FOR EMCOMPOSITION
            #                            SHOULD PROXY / DIRECT_PROJ BE IN DEPENDENCY_DICT?
            while entry in dependency_dict:
                # Prevent cycle from recurrent pathway
                if entry in pathway:
                    break
                pathway.insert(0, entry)
                entry = dependency_dict[entry]
                assert True
            pathway.insert(0, entry)
            # Only allow odd number of components since there must be one fewer Projections than Mechanisms
            assert len(pathway) % 2, \
                f"PROGRAM ERROR: There are one too many Projections in pathway: {' ,'.join(pathway)}"
            # Replace (port,index) tuples with nodes for actual pathway
            return [e if isinstance(e, MappingProjection) else e[0].owner for e in pathway]

        # breadth-first search starting with input node
        while len(queue) > 0:
            node, afferent_proj, current_comp = queue.popleft()

            # node is nested Composition that is an INPUT node of the immediate outer Composition,
            #   so put that in queue for procsssing in next pass through while loop
            if (isinstance(node, Composition) and node is not self
                    and any(isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                            for proj in node.afferents)):
                for output_port in node.input_CIM.output_ports:
                    for proj in output_port.efferents:
                        queue.append((proj.receiver.owner, proj, node))
                continue

            # node is output_CIM of outer Composition (i.e., end of pathway) which shouldn't happen yet
            if isinstance(node, CompositionInterfaceMechanism) and node is self.output_CIM:
                assert False, (f"PROGRAM ERROR: 'Got to output_CIM of outermost Composition '({self.name})' "
                               f"without detecting OUTPUT NODE at end of pathway")

            # End of pathway: OUTPUT Node of outer Composition
            if current_comp == self and (node in current_comp.get_nodes_by_role(NodeRole.OUTPUT)
                                         or not node.efferents):
                pathways.append(create_pathway(current_comp, node, afferent_proj))
                continue

            # # Get all efferent Projections of node,
            # #   including direct projections out of a nested Composition implemented in PyTorchCompositionWrapper
            efferent_projs = [(p, p.receiver.owner) for p in node.efferents if p in current_comp.projections]
            if not efferent_projs:
                efferent_projs = [(p, p.receiver.owner) for p in node.efferents
                                  if (p in current_comp._pytorch_projections
                                      or isinstance(p.receiver.owner, LossMechanism))]

            # Follow efferent Projection to next Node in pathway
            for efferent_proj, rcvr_mech in efferent_projs:
                # Ignore efferent Projections that do not have a learnable attribute
                #   or are ModulatoryProjections (i.e., including LearningProjections)
                # Note: if learnable==False, it will be passed along to PyTorch in PytorchProjectionWrapper
                if not hasattr(efferent_proj,'learnable') or isinstance(efferent_proj,ModulatoryProjection_Base):
                    continue

                # Deal with Projections to/from CIMs since nested comps can be learned in PyTorch mode
                if isinstance(rcvr_mech, CompositionInterfaceMechanism):

                    # Projection to input_CIM of a nested Composition
                    if rcvr_mech == rcvr_mech.composition.input_CIM:
                        assert rcvr_mech.composition is not current_comp
                        rcvr_comp = rcvr_mech.composition
                        # Get Node(s) in inner Composition to which Node projects (via input_CIM)
                        receivers = rcvr_mech._get_destination_info_from_input_CIM(efferent_proj.receiver)
                        for _, nested_rcvr, _ in [receivers] if isinstance(receivers, tuple) else receivers:
                            if rcvr_comp._input_comp_nodes_to_pytorch_nodes_map:
                                # If nested comp has _input_comp_nodes_to_pytorch_nodes_map, get nested_rcvr from it
                                nested_rcvr = rcvr_comp._input_comp_nodes_to_pytorch_nodes_map[nested_rcvr.input_port]
                            else:
                                # Otherwise, ensure that nested_rcvr is an INPUT Node of rcvr_comp
                                assert nested_rcvr in rcvr_comp.get_nodes_by_role(NodeRole.INPUT), \
                                    f"PROGRAM ERROR: '{nested_rcvr.name}' is not an INPUT Node of '{rcvr_comp.name}'"
                                # Assign efferent_proj (Projection to input_CIM) since it should be learned in PyTorch mode
                            rcvr_comp._add_dependency(afferent_proj, node, efferent_proj, nested_rcvr,
                                                      dependency_dict, queue, rcvr_comp)

                    # rcvr_mech is Nested Composition output_CIM:
                    # Projection is to output_CIM exiting from a nested Composition
                    elif rcvr_mech == current_comp.output_CIM and current_comp is not self:

                        # Get output_CIM info for current efferent_proj
                        output_CIM_input_port = efferent_proj.receiver
                        output_CIM = output_CIM_input_port.owner
                        # Get port of output_CIM that efferent_proj sends to, for use in findings its receiver(s) below
                        if efferent_proj in current_comp.projections:
                            output_CIM_output_port = output_CIM.port_map[efferent_proj.sender][1]
                        elif efferent_proj in current_comp._pytorch_projections:
                            output_CIM_output_port = \
                                (output_CIM.port_map)[efferent_proj.receiver.path_afferents[0].sender][1]

                        # Get all Node(s) in outer Composition to which node projects (via output_CIM)
                        receivers = rcvr_mech._get_destination_info_for_output_CIM(output_CIM_output_port)
                        # Replace efferent_proj(s) with one(s) from output_CIM to rcvr_mech(s) in outer Composition,
                        #   since that(those) is(are) the one(s) that should be learned in PyTorch mode
                        # Note:  _get_destination_info_for_output_CIM returns list of destinations
                        #        in order of output_CIM.output_port.efferents
                        if receivers:
                            for efferent_idx, receiver in enumerate(receivers):
                                if receiver:
                                    _, rcvr_mech, rcvr_comp = receiver
                                    assert rcvr_comp is not current_comp
                                efferent_proj = output_CIM_output_port.efferents[efferent_idx]
                                rcvr_comp._add_dependency(afferent_proj, node, efferent_proj, rcvr_mech,
                                                          dependency_dict, queue, rcvr_comp)
                        else:
                            # BREADCRUMB: NEED TO PASS afferent Projection to node here
                            pathways.append(create_pathway(current_comp, node, afferent_proj))

                    # rcvr_mech is Outermost Composition output_CIM:
                    # End of pathway: Direct projection from output_CIM of nested comp to outer comp's output_CIM
                    elif rcvr_mech is self.output_CIM:
                        # Assign node that projects to current node as OUTPUT Node for pathway
                        node_output_port = efferent_proj.sender
                        _, sender, _ = node._get_source_info_from_output_CIM(node_output_port)
                        pathway = create_pathway(current_comp, node, afferent_proj)
                        if pathway:
                            queue.popleft()
                            pathways.append(pathway)

                    else:
                        assert False, f"PROGRAM ERROR:  Unrecognized CompositionInterfaceMechanism: {rcvr_mech}"

                else:
                    if rcvr_mech in current_comp.nodes:
                        # rcvr_mech is still in nested Composition, so keep traversing that
                        current_comp._add_dependency(afferent_proj, node, efferent_proj, rcvr_mech,
                                                     dependency_dict, queue, current_comp)
                        current_comp._pytorch_projections.append(efferent_proj)
                        continue
                    elif rcvr_mech in self.nodes:
                        # rcvr_mech is in outer Composition (presumably a direct Pytorch Projection out of nested comp)
                        self._add_dependency(afferent_proj, node, efferent_proj, rcvr_mech,
                                             dependency_dict, queue, self)
                        continue
                    else:
                        assert False, \
                            (f"PROGRAM ERROR:  Unrecognized receiver ('{rcvr_mech.name}') of Projection from '{node.name}'.")

        return pathways

    @property
    def _has_learnable_pathways(self):
        return any(self._mech_is_receiver_in_learnable_pathway(port) for node in self.nodes for port in node.output_ports)

    def _mech_is_receiver_in_learnable_pathway(self, mech_output_port: OutputPort) -> bool:
        """Return True if `mech` receives a Projection from any pathway that has at least one learnable Projection"""
        mech = mech_output_port.owner
        if isinstance(mech, CompositionInterfaceMechanism):
            # Restrict search to afferent of input_port paired with mech_output_port
            afferents = mech._get_input_port_for_output_port(mech_output_port).path_afferents
        else:
            afferents = mech.path_afferents
        for afferent in afferents:
            if afferent.learnable:
                return True
            check_afferent_pathway = self._mech_is_receiver_in_learnable_pathway(afferent.sender)
            if check_afferent_pathway:
                return True
        return False

    def _mech_is_sender_in_learnable_pathway(self, sender: Union[Mechanism, InputPort]) -> bool:
        """Return True if `sender` sends a Projection to any pathway that ends in a LossMechanism."""
        assert not isinstance(sender, CompositionInterfaceMechanism), \
            "PROGRAM ERROR: Should not send CIM; use its input_port, relevant efferent is not otherwise resolvable."
        mech = sender if isinstance(sender, Mechanism) else sender.owner
        if isinstance(mech, CompositionInterfaceMechanism):
            assert mech is mech.composition.output_CIM,\
                f"PROGRAM ERROR: Should only reach output_CIMs from this method."
            destinations = mech._get_destination_info_for_output_CIM(sender)
            if destinations and any(isinstance(destination[0], LossMechanism) for destination in destinations):
                return True
        for efferent in mech.efferents:
            if isinstance(efferent.receiver.owner, LossMechanism):
                return True
            if isinstance(efferent.receiver.owner, LearningMechanism):
                continue
            sender = efferent.receiver if isinstance(efferent.receiver.owner, CompositionInterfaceMechanism) \
                else efferent.receiver.owner
            check_efferent_pathway = self._mech_is_sender_in_learnable_pathway(sender)
            if check_efferent_pathway:
                return True
        return False

    def _check_if_sample_is_in_learnable_pathway(self,
                                                 sample_port:OutputPort,
                                                 # # MODIFIED TEACHER_TARGET OLD:
                                                 # target_mech:ProcessingMechanism_Base=None,
                                                 # MODIFIED TEACHER_TARGET NEW:
                                                 target_spec=None,
                                                 # MODIFIED TEACHER_TARGET END
                                                 loss_mech:LossMechanism=None,
                                                 constructed_target_mechs:list=None,
                                                 action:Optional[Union[Literal[ERROR, WARNING]]]=None)->bool:
        """Take specified action if sample_port's owner has no afferent pathways with any learnable Projections.
        - target_spec argument is used to determine error_message;
        - if no action is specified, return True or False
        """
        if self._mech_is_receiver_in_learnable_pathway(sample_port):
            return True
        # SAMPLE is not in learnable pathway, so construct relevant error/warning message
        sample_mech = sample_port.owner
        if sample_mech not in self._get_all_nodes():
            # The erroneous SAMPLE specification will be caught and an error raised in _validate_loss_mech_specs()
            return False
        elif target_spec:
            # target was specified in *targets* arg of constructor
            if isinstance(target_spec, LossMechanism):
                target_msg = f"A LossMechanism ('{target_spec.name}')"
            elif (target_spec == TARGET or
                  (constructed_target_mechs and target_spec in constructed_target_mechs)):
                target_msg = "An external TARGET input"
            else:
                target_msg = f"A TARGET node ('{target_spec.full_name}')"
            error_msg = (f"{target_msg} can't be assigned to '{sample_mech.name}' in the 'targets' argument of "
                         f"'{self.name}', since there are no learnable Projections in any of the pathways that "
                         f"project to that Node.")
        else:
            if sample_mech in self.node_roles_mgr.get_nested_nodes_by_roles_at_any_level(self, NodeRole.SINGLETON):
                # Singletons are caught here because they are identified as OUTPUT Nodes,
                #   but are not specified in targets dict of learn() method.
                # Allow construction, as they could be a Mechanism for a learnable PyTorch module (e.g., GRU),
                #   and warning about potential non-learnability is handled in _instantiate_optimizer()
                # return False
                return SINGLETON
            # TARGET Nodes being constructed for all OUTPUT Nodes, so all must be in learnable pathways
            error_msg = (f"A target value is specified for '{sample_mech.name}' in the learn() method of "
                         f"'{self.name}', but that Node has no afferent pathways with any learnable Projections.")

        # Take specified action
        if action is ERROR:
            raise AutodiffCompositionError(error_msg)
        elif action is WARNING:
            warnings.warn(error_msg)
        return False

    def _check_if_target_is_in_sample_pathway(self,
                                              sample_port:OutputPort,
                                              target_port:OutputPort,
                                              pathways:list,
                                              context:Context):
        """Determine if target appears before the sample in any pathway.
        Returns True if target appears before sample in any pathway, False otherwise.
        """
        sample_mech = sample_port.owner
        target_mech = target_port.owner

        # Check each pathway for the sample
        for pathway in pathways:
            if sample_mech not in pathway:
                continue

            # Find positions of sample and target in this pathway
            sample_idx = next((i for i, node in enumerate(pathway) if node == sample_mech), None)
            target_idx = next((i for i, node in enumerate(pathway) if node == target_mech), None)

            # Warn if target comes before sample in the same pathway, and there is no learnable Projection between them
            # X TEST DONE
            warning_msg = (f"The target ({target_mech.name}) specified for a SAMPLE ({sample_mech.name}) in the "
                           f"'targets' argument of the constructor for '{self.name}' appears before it in the same "
                           f"pathway")
            if (target_idx is not None and sample_idx is not None and target_idx < sample_idx
                    and not self._warned_about_target_before_sample_in_pathway):
                if not any(isinstance(p, MappingProjection) and p.learnable for p in pathway[target_idx:sample_idx]):
                    warnings.warn(warning_msg + f"without a learnable Projection between them)")
                else:
                    warnings.warn(warning_msg + '; this may cause instabilities in learning.')
                self._warned_about_target_before_sample_in_pathway = True

    def _instantiate_loss_components(self, pathways, context, base_context):
        """Instantiate sample:target pairs, LossMechanisms, and any TARGET Nodes needed

        Overivew:
        - Use any specifications in self.targets (from **targets** arg of AutodiffComposition constructor)
            to identify sample-target pairs, and constuct LossMechanisms and any needed TARGET Nodes
        - If there are no specifications in self.targets, then use OUTPUT Nodes of pathways as samples
            and construct TARGET Nodes for each.

        Procedure:
        1) Handle specifications from constructor (in self.targets) in call to _instantiate_constructor_targets_args():
            - identifies sample-target pairs:
               - places them in self.sample_port_to_target_port_map
               - returns them as first item, placed in loss_mech_specs
            - creates TARGET Nodes (that receive external input) for any targets specified using TARGET keyword
              - returns them as second item, placed in target_mechs
        2) If there are no constructor specifications, then call _instantiate_default_targets():
            - assigns all OUTPUT Nodes of pathways as samples and TARGET Nodes as targets
                  this allows:
                    - external targets to be specified in learn() in the same way as for other execution_modes:
                        learn(targets = {<OUTPUT Node> : <value>}) -> TARGET Node
                        (mapping is done in _map_external_target_values_to_target_nodes()
                    - trial-by-trial losses to be kept aligned with inputs in batch / minibatch construction
                    - losses to be tracked for logging (as mechs of a Composition)
              - places them in self.sample_port_to_target_port_map
              - returns them as first item, placed in loss_mech_specs
           - creates any TARGET Nodes that have not yet been constructed
              - returns them as second item, placed in target_mechs
        3) Validate loss_mech_specs
        4) Use loss_mechs and target_mechs to instantiate LossMechanisms in call to _instantiate_loss_mechanisms():
           - constructs self.loss_mechs_map: {<LossMechanism>: (sample OutputPort, target OutputPort)}
           - adds LossMechanisms to AutodiffComposition
        5) Exclude LossMechanisms and TARGET Nodes from OUTPUT role and suppress warnings about role assignments
        """
        # If loss_components have already been instantiated, skip
        if self._sample_target_pairs and any(isinstance(node, LossMechanism) for node in self.nodes):
            return

        context = Context(source=ContextFlags.METHOD, execution_id=context.execution_id)

        if self.targets:
            # Instantiate any sample-target specifications in **targets** arg of AutodiffComposition constructor
            self._instantiate_constructor_targets_args(pathways, context, base_context)
        else:
            # No target specifications in constructor, so instantiate default TARGET Node assignments,
            self._instantiate_default_targets(pathways, context, base_context)

        loss_mech_specs = [(spec.sample_port, spec.target_port) for spec in self._sample_target_pairs]
        target_mechs = [spec.target_mech for spec in self._sample_target_pairs]
        self._validate_loss_mech_specs(loss_mech_specs, context)
        self._parse_constructor_targets_specs()
        loss_mechs = self._instantiate_loss_mechanisms(loss_mech_specs, context, base_context)

        # Exclude LossMechanisms and TARGET Nodes from OUTPUT role and suppress warnings about role assignments
        for mech in loss_mechs + target_mechs:
            self.exclude_node_roles(mech, NodeRole.OUTPUT, context=context)
            for output_port in mech.output_ports:
                output_port.parameters.require_projection_in_composition.set(False, override=True)

        # Error if there are any learnable Projections in pathways that do not end with a LossMechanism
        self._check_for_orphaned_learnable_projections(pathways)

    def _check_for_orphaned_learnable_projections(self, pathways):

        orphaned_learnable_projections = []
        for pathway in pathways:
            backwards_pathway = pathway[::-1]
            # BREADCRUMB: DO ALL OF THIS USING pytorch_represetation TO AVOID ISSUES WITH NESTED COMPS AND CIMs
            # Only bother with a pathway if it has a learnable Projection
            if any(proj.learnable for proj in pathway if isinstance(proj, MappingProjection)):
                # BREADCRUMB: SHOULD REPLACE THIS WITH METHOD THAT GETS ALL TERMINAL NODES FOR EFFERENTS OF A MECHANISM
                pathway_orphans = [] # Keep these separate so their order be restored (re-reversed) within the pathway
                for i, item in enumerate(backwards_pathway):
                    if isinstance(item, LossMechanism):
                        break
                    # If item is a Node and any of its efferents project to a LossMechanism, break
                    if isinstance(item, ProcessingMechanism_Base):
                        sender = backwards_pathway[i - 1].receiver if isinstance(item, CompositionInterfaceMechanism) else item
                        if self._mech_is_sender_in_learnable_pathway(sender):
                            break
                    if isinstance(item, MappingProjection) and item.learnable:
                        pathway_orphans.append(item)
                orphaned_learnable_projections.extend(pathway_orphans[::-1])
        if orphaned_learnable_projections:
            bad_projs_names = [f"'{p}'" for p in orphaned_learnable_projections]
            plural = len(orphaned_learnable_projections) > 1
            s = 's' if plural else ''
            is_are = 'are' if plural else 'is'
            do_does = 'do' if plural else 'does'
            a_not_a = '' if plural else 'a '
            # X TEST DONE
            raise AutodiffCompositionError(f"The following Projection{s} {is_are} learnable but {is_are} in {a_not_a}"
                                           f"pathway{s} that {do_does} not end in a LossMechanism, and therefore "
                                           f"cannot be learned: {', '.join(bad_projs_names)}. Reminder: when *any* "
                                           f"targets are specified in the 'targets' argument of the constructor for "
                                           f"an AutodiffComposition, then ones must be specified for *all* learnable "
                                           f"pathways in that Composition; if none are specified, the OUTPUT Nodes of "
                                           f"all learnable pathways in that Composition are treated as SAMPLEs, and "
                                           f"TARGET Nodes are automatically constructed for each, for which inputs "
                                           f"must then be provided in the 'targets' argument of the learn() method "
                                           f"when it is called.")

    def _instantiate_constructor_targets_args(self, pathways, context, base_context):
        """Instantiate targets specified by user in **targets** argument of AutodiffComposition constructor
        - These may be in
            -  target attribute of an explicitly specified LossMechanism
            -  a (sample:target) tuple
            -  a list containing tuples and/or LossMechanisms
            -  or dict of {sample:target} pairs
          where:
              sample = OutputPort or ProcessingMechanism,
              target = OutputPort, ProcessingMechanism, or TARGET keyword
        - Identify all samples and assign NodeRole.SAMPLE to them
        - Instantiate TARGET Nodes for any targets specified as TARGET, and assign NodeRole.TARGET
        - Update self._sample_target_pairs (with SAMPLE and TARGET Mechanisms and OutputPorts)
        """
        # BREADCRUMB: PRESERVE SPEC RATHER THAN FULLY CONVERTING TO PORT (FOR _sample_target_specs())

        loss_mech_specs = []
        target_mechs = []
        constructed_target_mechs = []

        for loss_mech_spec in self.targets:
            if isinstance(loss_mech_spec, LossMechanism):
                sample_port = loss_mech_spec.sample
                sample_mech = sample_port.owner
                target_port = loss_mech_spec.target
                target_mech = target_port.owner
                # If sample specified for LossMechanism is not in a pathway with at least one learnable Projection
                #   then raise error, as executing its LossFunction in pytorch will cause a crash
                self._check_if_sample_is_in_learnable_pathway(sample_port=sample_port,
                                                              target_spec=target_mech,
                                                              loss_mech=loss_mech_spec,
                                                              constructed_target_mechs=constructed_target_mechs,
                                                              action=ERROR)
                # MODIFIED TEACHER_TARGET NEW:
                sample_spec = target_spec = loss_mech_spec
                # MODIFIED TEACHER_TARGET END

            elif isinstance(loss_mech_spec, tuple):
                sample_spec, target_spec = loss_mech_spec
                _parse_spec = lambda spec : ((spec, spec.owner) if isinstance(spec, OutputPort)
                                             else (spec.output_port, spec))
                sample_port, sample_mech = _parse_spec(sample_spec)
                target_port, target_mech = _parse_spec(target_spec) if target_spec is not TARGET else (None, None)

                # If specified sample Mechanism is not in a pathway with at least one learnable Projection
                #   then raise error, as constructing a LossMechanism with a LossFunction that tries to compute
                #   loss in pytorch will cause a crash
                _learnable = self._check_if_sample_is_in_learnable_pathway(sample_port=sample_port,
                                                                           # # MODIFIED TEACHER_TARGET OLD:
                                                                           # target_mech=target_mech,
                                                                           # MODIFIED TEACHER_TARGET NEW:
                                                                           target_spec=target_spec,
                                                                           # MODIFIED TEACHER_TARGET END
                                                                           loss_mech=None,
                                                                           constructed_target_mechs=None,
                                                                           action=ERROR)
                # Determine whether target is internal node or TARGET keyword
                if isinstance(target_spec, (OutputPort, ProcessingMechanism_Base)):
                    # target is internal Node
                    self._check_if_target_is_in_sample_pathway(sample_port, target_port, pathways, context)
                elif target_spec == TARGET:
                    # target is TARGET keyword, so construct TARGET Node
                    if sample_port in self.sample_port_to_target_port_map:
                        # TARGET Node has already been constructed for specified sample Port
                        continue
                    sample_name = (sample_port.full_name if len(sample_port.owner.output_ports)>1
                                   else sample_port.owner.name)
                    # TEACHER_TARGET BREADCRUMB: SHOULD THIS BE FOR sample_port NOT sample_mech?
                    target_mech = ProcessingMechanism(default_variable = np.array([np.zeros_like(value) for value
                                                                                   in sample_mech.value],
                                                                                  dtype=object),
                                                      name= 'TARGET for ' + sample_name)
                    target_mech._initialize_from_context(context, base_context, override=False)
                    target_port = target_mech.output_port
                    self.add_node(target_mech, required_roles=[NodeRole.TARGET, NodeRole.INPUT], context=context)
                    constructed_target_mechs.append(target_mech)
                else:
                    assert False, (f"PROGRAM_ERROR: unrecognized value of target specification "
                                   f"({loss_mech_spec[1]} for '{self.name}'.")
            else:
                assert False, (f"PROGRAM_ERROR: unrecognized specification for self.targets "
                               f"({loss_mech_spec} for '{self.name}'.")

            self._sample_target_pairs.append(SampleTargetPair(sample_port.owner,
                                                              sample_port,
                                                              target_mech,
                                                              target_port))
            self._sample_target_specs.append(SampleTargetSpec(sample_port, sample_spec,
                                                              target_port, target_spec, None,
                                                              CONSTRUCTOR_TARGETS))

            self.require_node_roles(sample_mech, NodeRole.SAMPLE, context=context)

        self._validate_constructor_targets_specs()

    def _instantiate_default_targets(self, pathways: list, context, base_context) -> Tuple[List, List]:
        """Construct default TARGET Nodes (since none were specified in **targets** arg of constructor
        Current default is to treat all OUTPUT Nodes as samples, and assign them TARGET Nodes
        IMPLEMENTATION NOTE:
           This is to support legacy behavior, in which targets are not specified explicitly
        - Only add TARGET Nodes if *not* already present in self.sample_port_to_target_port_map.values(),
           to avoid duplication in multiple calls, including from command line
           (see test_xor_training_identicalness_standard_composition_vs_PyTorch_and_LLVM for example)
        - Update self.sample_port_to_target_port_map with construted TARGET Nodes
        - Add constructed TARGET Nodes to AutodiffComposition with NodeRole.TARGET and NodeRole.INPUT
        Return list of loss_mech_specs ((sample OutputPort, targetOutputPort) tuples) and constructed TARGET Nodes
        """
        pathway_terminal_nodes = [mech for mech in [pathway[-1] for pathway in pathways]]
        identified_output_nodes = self._identify_output_nodes(context)
        output_ports_for_learning = []
        constructed_target_mechs = []
        for node in [n for n in identified_output_nodes if n in pathway_terminal_nodes]:
            output_ports_for_learning.extend(node.output_ports)
        target_mechs = self.get_nodes_by_role(NodeRole.TARGET)
        for output_port_for_learning in output_ports_for_learning.copy():
            _learnable = self._check_if_sample_is_in_learnable_pathway(sample_port=output_port_for_learning,
                                                                       target_spec=None,
                                                                       loss_mech=None,
                                                                       constructed_target_mechs=constructed_target_mechs,
                                                                       action=ERROR)
            # If no error is generated in sample_is_in_learnable_pathway(), sample is a singeton;
            #   warning about non-learnability is handled in _instantiate_optimizer()
            if _learnable is SINGLETON or _learnable is False:
                output_ports_for_learning.remove(output_port_for_learning)
                continue
            # Check for existing TARGET Nodes
            existing_output_ports_for_learnings = [sample for sample, target in self.loss_mechs_map.values()]
            # Get or construct TARGET Node if none exists for OUTPUT Node
            if output_port_for_learning not in existing_output_ports_for_learnings:
                # Check that TARGET Node doesn't already exist for OUTPUT Node
                #    (may have been created for PNL learning in call to add_backpropagation_learning_pathway)
                existing_comparators = [mech for mech in self.nodes if
                                        isinstance(mech, ComparatorMechanism) and
                                        NodeRole.LEARNING_OBJECTIVE in self.get_roles_by_node(mech)]
                comparators_for_output_port = [mech for mech in existing_comparators
                               if mech.input_ports[SAMPLE].path_afferents[0].sender is output_port_for_learning]
                assert len(comparators_for_output_port) <= 1, (f"PROGRAM ERROR: multiple ComparatorMechanisms found "
                                               f"for '{output_port_for_learning.full_name}' in {self.name}'.")
                if comparators_for_output_port:
                    target_mech = comparators_for_output_port[0].input_ports[TARGET].path_afferents[0].sender.owner
                    # Autodiff now owns this TARGET Node, so dissociate from learning_components used for Python
                    self.exclude_node_roles(target_mech, [NodeRole.LEARNING], context=context)
                    # TARGET Node already exists, so no need to construct
                    continue
                else:
                    sample = output_port_for_learning
                    sample_name = sample.full_name if len(sample.owner.output_ports)>1 else sample.owner.name
                    target_mech = ProcessingMechanism(default_variable = np.array([np.zeros_like(value)
                                                                                   for value in output_port_for_learning.value],
                                                                                  dtype=object),
                                                      name= f"{TARGET} for " + sample_name)
                    target_mech._initialize_from_context(context, base_context, override=False)
                    constructed_target_mechs.append(target_mech)
                target_mechs.append(target_mech)
                # TEACHER_TARGET BREADCRUMB: require_node_roles() NEEDS TO HANDLE SCOPE, TO ASSIGN TO NESTED NODES
                self.require_node_roles(output_port_for_learning.owner, NodeRole.SAMPLE, ALL, context)
        loss_mech_specs = list(zip(output_ports_for_learning, [target.output_port for target in target_mechs]))
        assert len(output_ports_for_learning) == len(target_mechs), \
            f"PROGRAM_ERROR: Number of output_ports_for_learning is not same as number of target_mechs constructed."
        self._sample_target_pairs.extend([SampleTargetPair(s.owner, s, t, t.output_port)
                                          for s, t in zip(output_ports_for_learning, target_mechs)])

        self.add_nodes(target_mechs, required_roles=[NodeRole.TARGET, NodeRole.INPUT], context=context)
        return loss_mech_specs, target_mechs

    def _validate_loss_mech_specs(self, loss_mech_specs: list, context) -> Tuple[List, List]:
        """Validate specifications used to construct LossMechanism in _instantiate_loss_components"""
        if not loss_mech_specs:
            if context.execution_id:
                # Raise error on attempt to learn without any learnable Projections
                raise AutodiffCompositionError(f"Learning cannot be executed for '{self.name}' "
                                               f"since it does not have any learnable Projections.")
            else:
                # Raise warning on attempt to construct without any learnable Projections
                if not self._warned_about_no_learnable_projections:
                    warnings.warn(f"It will not be possible to execute learning for '{self.name}' "
                                  f"since it does not have any learnable Projections.")
                self._warned_about_no_learnable_projections = True

        spec_as_mech = lambda spec : spec.owner if isinstance(spec, OutputPort) else spec
        nodes_in_comp = self._get_all_nodes()
        bad_samples = []
        bad_targets = []
        for loss_mech_spec in list(loss_mech_specs):
            # Assume that self.targets is a list of LossMechanisms and/or tuples specifying sample:target pairs
            assert isinstance(loss_mech_spec, (LossMechanism, tuple)), \
                (f"PROGRAM ERROR: item in self.targets is neither LossMechanism nor 2-item tuple: {loss_mech_spec};"
                 f"should have been caught in targets Parameter validation.")
            if isinstance(loss_mech_spec, tuple):
                assert (len(loss_mech_spec) == 2
                        and all((isinstance(item, OutputPort) or item==TARGET) for item in loss_mech_spec)), \
                    (f"PROGRAM ERROR: tuple in self.targets either doesn't have two items "
                     f"or one is not a Mechanisms: {loss_mech_spec}; "
                     f"should have been caught in targets Parameter validation.")
                if spec_as_mech(loss_mech_spec[0]) not in nodes_in_comp:
                    bad_samples.append(loss_mech_spec[0])
                if spec_as_mech(loss_mech_spec[1]) not in nodes_in_comp:
                    bad_targets.append(loss_mech_spec[1])

            else:
                assert False, (f"PROGRAM ERROR: unrecognized item in self.targets: {item}")

        # X TEST DONE
        if bad_samples or bad_targets:
            bad_samples_str = ', '.join([f"'{spec.full_name}'" for spec in bad_samples])
            bad_targets_str = ', '.join([f"'{spec.full_name}'" for spec in bad_targets])
            plural = len(bad_samples) > 1
            a = 'a ' if not plural else ''
            s = 's' if plural else ''
            bad_samples_msg = f"{a}sample{s} ({bad_samples_str})" if bad_samples_str else ""
            plural = len(bad_targets) > 1
            a = 'a ' if not plural else ''
            s = 's' if plural else ''
            are_is = 'are' if plural else 'is'
            bad_targets_msg = f"{a}target{s} ({bad_targets_str})" if bad_targets_str else ""
            plural = bad_samples and bad_targets
            both = ' and ' if plural else ''
            error_msg = (f"The specification for the 'targets' argument of the constructor for '{self.name}' "
                         f"contains {bad_samples_msg}{both}{bad_targets_msg} that {are_is} not in the Composition.")


    def _instantiate_loss_mechanisms(self, loss_mech_specs:list, context, base_context)->list:
        """Construct and/or add LossMechanisms (and their MappingProjections) to AutodiffComposition
        - loss_mech_specs is a list with (sample OutputPort, target OutputPort) tuples and/or LossMechanisms
        - If item is a (sample OutputPort, target OutputPort) tuple construct LossMechanism with:
             LossMechanism.input_port[SAMPLE] and LossMechanism.sample = sample OutputPort
             LossMechanism.input_port[TARGET] and LossMechanism.target = target OutputPort
             LossMechanism.loss = self.loss_spec
        - Add LossMechanisms to AutodiffComposition, with NodeRole.LEARNING_OBJECTIVE
        - Assign self.loss_mechs_map as {<LossMechanism>: (sample OutputPort, target OutputPort)}
        Return list of constructed LossMechanisms
        """
        for i, loss_mech_spec in enumerate(loss_mech_specs):
            if isinstance(loss_mech_spec, LossMechanism):
                sample = loss_mech_spec.sample
                target = loss_mech_spec.target
                loss_mech = loss_mech_spec
            elif isinstance(loss_mech_spec, tuple):
                sample, target = loss_mech_spec
                # Get loss_mech for sample if it already has one
                loss_mechs_for_sample = [l for l, sample_and_target in self.loss_mechs_map.items()
                                          if sample in sample_and_target]
                if loss_mechs_for_sample:
                    if len(loss_mechs_for_sample) > 1:
                        errant_target_names = [f"'{mech.target.full_name}'" for mech in loss_mechs_for_sample]
                        raise AutodiffCompositionError(
                            f"'{sample.full_name}' is associated with more than one TARGET in '{self.name}: "
                            f"{' ,'.join(errant_target_names)}.")
                    else:
                        loss_mech = loss_mechs_for_sample[0]
                        continue
                else:
                # if not any(sample in sample_and_target for sample_and_target in self.loss_mechs_map.values()):
                # Construct LossMechanism
                    # If there is no loss_mech for the current sample, instantiate one
                    # IMPLEMENTATION NOTE:
                    #        Don't allow multiple LossMechanisms to train the same SAMPLE Node
                    #        But it IS OK to have multiple LossMechanisms use the same TARGET Node
                    #        (i.e., to train multiple SAMPLES)
                    name = sample.full_name if len(sample.owner.output_ports)>1 else sample.owner.name
                    loss_mech = LossMechanism(name=f"LOSS for {name}",
                                              sample=sample,
                                              target=self.sample_port_to_target_port_map[sample],
                                              function=None,
                                              loss=self.loss_spec)
                    loss_mech._initialize_from_context(context, base_context, override=False)
                    for proj in loss_mech.path_afferents:
                        proj.learnable= False
            else:
                assert False, f"PROGRAM ERROR: loss_mech_spec should have been a LossMechanism or tuple by now."

            for proj in loss_mech.path_afferents:
                # TEACHER_TARGET BREADCRUMB: REVISE BELOW TO ENFORCE THESE ON CONSTRUCTION in LossMechanism
                # IMPLEMENTATION NOTE:
                #     This is checked here because the Projections to the LossMechanism
                #     are constructed by reference to its afferents (sample and target)
                assert is_identity_matrix(proj.parameters.matrix.get()), \
                    (f"PROGRAM ERROR: Matrix of projection to LossMechanism "
                     f"('{proj.name}') is not an identity matrix. ")
                assert proj.learnable is False, (f"PROGRAM ERROR: The 'learnable' attribute of a projection to a "
                                                 f"LossMechanism ('{proj.name}') is not False")

            self.loss_mechs_map[loss_mech] = (sample, target)

        # Add LossMechanisms to AutodiffComposition, with required NodeRoles
        loss_mechs = list(self.loss_mechs_map.keys())
        self.add_nodes(loss_mechs, required_roles=[NodeRole.LEARNING_OBJECTIVE], context=context)
        return loss_mechs

    def _add_dependency(self,
                        afferent_proj:MappingProjection,
                        sender:ProcessingMechanism_Base,
                        projection:MappingProjection,
                        receiver:ProcessingMechanism_Base,
                        dependency_dict:dict,
                        queue:deque,
                        comp:Composition):
        """Append dependencies to dependency list, and next node to queue used in _get_pytorch_backprop_pathway().
        This uses the Projection from node (i.e., efferent of node) to receiver to implement the relevant dependencies
        for construcing the pathway; however, this can be overridden by a subclass of Autodiff to implement a custom
        pathway (see example in GRUComposition).

        **projection** is used to dereference **sender** and **receiver** afferents/efferents in the relevant port
        which are used in the dependency_dict, to prevent overwritting of entries that involve different ports of
        the same Mechanisms.

        add to dependency_dict = {projection: (sender_input_port, idx)
                                  receiver_input_port, idx) : projection
        """

        if afferent_proj is None:
            return

        if any(isinstance(mech, CompositionInterfaceMechanism) for mech in (sender, receiver)):
            # Should not be passed any CIMS (should have been handled in call from _get_pytorch_backprop_pathway
            assert False, f"PROGRAM ERROR: CIM unexpectedly encountered in {self.name}._add_dependency()"

        # Dereference InputPort of sender and index of its afferent
        sender_port = afferent_proj.receiver
        sender_idx = sender_port.path_afferents.index(afferent_proj)
        # MODIFIED TEACHER_TARGET NEW:
        if isinstance(sender_port.owner, CompositionInterfaceMechanism):
            sender_to_cim_info = sender_port.owner._get_source_node_for_input_CIM(sender_port)
            if sender_to_cim_info:
                _, sender_to_cim_mech, _ = sender_to_cim_info
                # Find direct Projection from sender_to_cim_mech to sender,
                direct_proj_to_sender_port = next((p for p in sender.path_afferents
                                                   if p.sender.owner == sender_to_cim_mech), None)
                if direct_proj_to_sender_port:
                    # Use that to dereference sender_port and sender_idx
                    sender_port = direct_proj_to_sender_port.receiver
                    sender_idx = sender.path_afferents.index(direct_proj_to_sender_port)
        # MODIFIED TEACHER_TARGET END

        # Dereference InputPort of receiver and index of its afferent
        if isinstance(projection.receiver.owner, CompositionInterfaceMechanism):
            # Get Projection from sender to receiver_port of input_CIM
            input_cim_output_port = projection.receiver.owner._get_output_port_for_input_port(projection.receiver)
            # Use that to dereference receiver_port
            proj_from_input_cim_to_receiver = input_cim_output_port.efferents[0]
            receiver_port = proj_from_input_cim_to_receiver.receiver
            proj_to_receiver = next((p for p in receiver_port.path_afferents
                                     if p.sender.owner == sender_port.owner),None)
            # Dereference receiver_idx
            if proj_to_receiver:
                receiver_idx = receiver_port.path_afferents.index(proj_to_receiver)
            else:
                proj_to_receiver = projection
                receiver_idx = receiver_port.path_afferents.index(proj_from_input_cim_to_receiver)
        else:
            proj_to_receiver = projection
            receiver_port = projection.receiver
            receiver_idx = receiver_port.path_afferents.index(projection)

        dependency_dict[(receiver_port, receiver_idx)] = proj_to_receiver
        dependency_dict[proj_to_receiver] = (sender_port, sender_idx)
        assert True

        queue.append((receiver_port.owner, projection, comp))

    def _validate_optimizer_params(
        self,
        opt_params: OptimizerParams,
        context: Context,
        err_source: str = '',
        runtime: bool = True,
    ):
        if runtime:
            for comp in [self] + self._get_nested_compositions():
                if not hasattr(comp, '_validate_optimizer_param_invalid_GRU_projections'):
                    continue

                for o_param in opt_params:
                    comp._validate_optimizer_param_invalid_GRU_projections(
                        o_param, err_source=f'{self}.learn()'
                    )

        super()._validate_optimizer_params(
            opt_params=opt_params,
            context=context,
            err_source=err_source,
            runtime=runtime,
        )

    # BREADCRUMB: move some of what's done in the methods below to a "_validate_params" type of method
    @handle_external_context(fallback_most_recent=True)
    def _build_pytorch_representation(self,
                                      learning_rate=None,
                                      optimizer_params=None,
                                      context=None,
                                      new=None,
                                      base_context=Context(execution_id=None)):
        """Build a Pytorch representation of the AutodiffComposition
        Construct PytorchCompositionWrapper that is used for learning in PyTorch, which is assigned to
        self.pytorch_representation.

        A new pytorch_representation is constructed if:
            self.pytorch_representation == None
            **new** is specified as True
        If _build_pytorch_representation() is called with **new**==None and a pytorch_representation already exists,
            a warning issued and the call is ignored.

        By default (learning_rate=None), the learning_rates specified in the **learning_rate**
        argument of the constructor for the Composition (and stored in self._learning_rates_dict) are
        used to construct the pytorch_representation. However:
        - if **learning_rate** is specified (in a call from the COMMANDLINE),
           that can be used to override the default values, as described under the learning_rate argument below;
        - if **optimizer_params** is specified (in a call from learn(), that is used to specify the learning_rates;
        - if **optimizer_params** is specified in a call from COMMAND_LINE, an error is returned.

        Arguments
        ---------

        new : bool or None : default None
            specifies creation of a new pytorch_representation, using optimizer
            parameter values from the constructor
            as the base values, and updated with any specified in the **learning_rates** arg.  If the method is called
            from the command_line more than once without **new** specified as `True`, warns and ignores.

        learning_rate : float, int, dict : default None
            if None, then the values from the constructor
            are used to assign learning_rates to all Projections in the Composition (and any nested within it)
            (see `Composition_Learning_Rate` for details of specification); if a numeric values is specified,
            that is used as the default learning_rate for the pytorch_representation (replacing
            composition.learning_rate); if a dict is specified, entries are moved to optmizer_params and replace
            values for the specified Projections as well as the Composition's learning_rate (if DEFAULT_LEARNING_RATE
            is specified in the dict).

            .. note::
               Projection-specific learning_rates specified in a dict assigned to **learning_rate** here, like
               any specified in the constructor for the Composition, are stored in the corresponding Projections'
               `learning_rate <MappingProjection.learning_rate>` Parameter under **context**.
        """
        optimizer_params = optimizer_params or {}
        if self.scheduler is None:
            self.scheduler = Scheduler(graph=self.graph_processing)

        # optimization parameters currently are used even if not building first time or rebuilding
        if learning_rate is not None:
            self.parameters.learning_rate.set(learning_rate, context)
            self.parameters.learning_rate._user_specified = True

        # Construct a new pytorch_representation if none exists or new is specified

        from psyneulink.core.llvm import ExecutionMode
        if self._has_learnable_pathways:
            self.infer_backpropagation_learning_pathways(execution_mode=ExecutionMode.PyTorch,
                                                         context=context,
                                                         base_context=base_context)
        else:
            # No learnable pathways
            if self.targets:
                # The **targets** arg was specified illegally in the constructor
                raise AutodiffCompositionError(f"The 'targets' argument was specified for '{self.name}', "
                                               f"but it has no learnable pathways")
            # No targets specified, so just warn that none can be.
            warnings.warn(f"No learnable pathways were found in '{self.name}'; therefore, "
                          f"no pytorch_representation will be constructed, and learning will not be possible.")

        if self.parameters.pytorch_representation._get(context=context) is None or new:
            # Instantiate pytorch_representation
            context.composition = self
            self.pytorch_composition_wrapper_type(composition=self,
                                                  device=self.device,
                                                  context=context,
                                                  base_context=base_context)
        elif context.flags & ContextFlags.COMMAND_LINE:
            warnings.warn(f"The '_build_pytorch_representation() method for '{self.name}' has already been called "
                          f"directly from the command line; this and any additional calls will be ignored. "
                          f"Make any desired modifications to parameters (e.g., learning_rates) either in the "
                          f"constructor for the AutodiffComposition, or its learn() method.")

        # Get pytorch_representation (assigned in constructor for PytorchCompositionWrapper)
        pytorch_rep = self.parameters.pytorch_representation._get(context)

        # BREADCRUMB: MOVE THIS TO PytorchCompositionWrapper __init__(), since it belongs to that
        # Set up optimizer
        old_opt = pytorch_rep.optimizer
        # Get default learning rate (used for all Parameters for which specific learning_rates are not specified),
        #    giving precedence to learning_rate specified in call to learn() (stored in self._runtime_learning_rate)
        #    over learning_rate specified in constructor (passed in above as learning_rate)
        if isinstance(learning_rate, dict):
            if optimizer_params:
                # if learning_rate is a dict, optimizer_params should not have been passed in call
                assert context.flags & ContextFlags.COMMAND_LINE, \
                    ("PROGRAM ERROR: 'optmizer_params' assigned when learning_rate assigned as a dict "
                     "in internal call to _build_pytorch_representation() for '{self.name}'.")
                assert False, \
                    ("PROGRAM ERROR:  Assignment of 'optimizer_params' in a direct call to "
                     "_build_pytorch_representation() from the command line is not currently supported.")

        if (old_opt is None or new) and new is not False:
            # Instantiate a new optimizer if there isn't one yet or new has been called and is not blocked)
            if context.runmode == ContextFlags.LEARNING_MODE:
                # If optimizer is being constructed de novo in call to learn(),
                #    instantiate it using params specified in constructor (if any) since:
                #   - need those implemented in a params_group to revert back to after execution of learn()
                #   - the ones in the call to learn() will be applied in call to _update_optimizer_params() below
                pytorch_rep.optimizer = self._instantiate_optimizer(context)
                # Then update optimizer params with any specified in the call to learn()
                self._update_optimizer_params(
                    pytorch_rep.optimizer,
                    Context(
                        source=ContextFlags.METHOD, runmode=context.runmode, execution_id=context.execution_id
                    ),
                )
            else:
                # Otherwise, if call is from Composition constructor, use params specified by user in that call
                pytorch_rep.optimizer = self._instantiate_optimizer(context)

        elif context.source is ContextFlags.SHOW_GRAPH:
            # Don't bother updating for call to show_graph()
            pass
        else:
            # Otherwise, just update it
            self._update_optimizer_params(old_opt,
                                                 Context(source=ContextFlags.METHOD,
                                                         runmode=context.runmode,
                                                         execution_id=context.execution_id))
        # Set up loss function
        if self.loss_function is not None:
            logger.warning("Overwriting 'loss_function' for AutodiffComposition {}! Old loss function: {}".format(
                self, self.loss_function))
        if callable(self.loss_spec):
            self.loss_function = self.loss_spec
        else:
            self.loss_function = self._get_loss(self.loss_spec)

        return pytorch_rep

    def _instantiate_optimizer(self, context):
        learning_rate = self._get_optimizer_param_value('learning_rate', context)
        composition_optimizer_params = OptimizerParams.from_component(self, context)
        self._validate_optimizer_params(composition_optimizer_params, context)
        try:
            runtime_optimizer_params = self.runtime_optimizer_params[context.execution_id]
        except KeyError:
            # no runtime for this context, ignore validation
            pass
        else:
            self._validate_optimizer_params(runtime_optimizer_params, context, 'the learn() method')

        if self.optimizer_type not in ['sgd', 'adam']:
            raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                           "Currently, Stochastic Gradient Descent and Adam are the only available "
                                           "optimizers (specified as 'sgd' or 'adam').")
        pytorch_rep = self.parameters.pytorch_representation._get(context)
        params = pytorch_rep.parameters()
        if (len(pytorch_rep.state_dict()) == 0):
            # avoid expiring params generator
            assert len(list(params)) == 0, \
                (f"PROGRAM ERROR: '{self.name}'.pytorch_representation has parameters "
                 f"but no learnable Projections or entries in its state_dict()")
            warnings.warn(f"'{self.name}' contains no Projections, so it has no params for Pytorch to learn.")
            return
        if self.optimizer_type == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=self.weight_decay)
        self._update_optimizer_params(optimizer, context, validate=False)
        return optimizer

    def _update_optimizer_params(self, optimizer, context, validate=True):
        if validate:
            composition_optimizer_params = OptimizerParams.from_component(self, context)
            self._validate_optimizer_params(composition_optimizer_params, context)
            try:
                runtime_optimizer_params = self.runtime_optimizer_params[context.execution_id]
            except KeyError:
                # no runtime for this context, ignore validation
                pass
            else:
                self._validate_optimizer_params(runtime_optimizer_params, context, 'the learn() method')

        pytorch_rep = self.parameters.pytorch_representation._get(context)
        pytorch_rep._update_optimizer_params(optimizer, context)

    def get_target_nodes(self, execution_mode=pnlvm.ExecutionMode.PyTorch,
                         context=None, base_context=None):
        """Return `TARGET` `Nodes <Composition_Nodes>` of the AutodiffComposition."""
        self.infer_backpropagation_learning_pathways(execution_mode=execution_mode,
                                                     context=context, base_context=base_context)
        return super(AutodiffComposition, self).get_target_nodes(context, base_context)

    def autodiff_forward(self,
                         inputs, targets,
                         optimization_num,
                         synch_with_pnl_options, retain_in_pnl_options,
                         execution_mode, scheduler, context):
        """Perform forward pass of model and compute loss for a batch of trials in Pytorch mode.
        COMMENT:
        ADD MENTION OF optimization steps here?
        COMMENT
        Losses are accumulated, and error is backpropagated by compositionrunner.run_learning()
        before the next time it calls run(), in a call to backward() by do_gradient_optimization()
        in _batch_inputs() or _batch_function_inputs(),

        Returns values of all OUTPUT Nodes of pytorch_representation
        """
        assert execution_mode is pnlvm.ExecutionMode.PyTorch
        pytorch_rep = self.parameters.pytorch_representation._get(context)

        # --------- Get current values of nodes  -------------------------------------------------

        # We need to pass both inputs and targets to the forward method in one dict, convert any numpy arrays to torch
        # tensors
        inputs_and_targets = {**inputs, **targets}
        for component, val in list(inputs_and_targets.items()):
            if isinstance(val, torch.Tensor):
                inputs_and_targets[component] = val.to(device=self.device, dtype=torch.double)
            else:
                inputs_and_targets[component] = torch.tensor(val, device=self.device, dtype=torch.double)

        # Execute PytorchCompositionWrapper to get value of all OUTPUT nodes for current trial
        output_values = pytorch_rep.forward(inputs=inputs_and_targets,
                                            optimization_num=optimization_num,
                                            synch_with_pnl_options=synch_with_pnl_options,
                                            retain_in_pnl_options=retain_in_pnl_options,
                                            full_sequence_mode=self.full_sequence_mode,
                                            sequence_lengths=(
                                                None if not hasattr(pytorch_rep, '_batch_seq_lengths')
                                                else pytorch_rep._batch_seq_lengths),
                                            context=context)


        pytorch_rep.minibatch_loss = self.compute_loss(targets, pytorch_rep, context)
        pytorch_rep.minibatch_loss_count += 1

        return output_values

    def compute_loss(self, targets, pytorch_rep, context):
        """Compute loss for each trial
        Can be overridden to use direct/dedicated/customized computation of loss by subclasses.
        IMPLEMENTATION NOTE:
            targets arg is included for overrides; LossMechanism uses its target input directly
        """
        return self.compute_loss_using_loss_mechanisms(targets, pytorch_rep, context)

    def compute_loss_using_loss_mechanisms(self, targets, pytorch_rep, context):
        """Compute loss after execution of autodiff_forward()
        Use values of LossMechanism(s) that computed loss for each pathway
        """
        trial_loss = 0
        assert self.loss_mechs_map, (f"PROGRAM ERROR: compute_loss() called for '{self.name} which does not "
                                     f"have any LossMechanism(s) or an override to compute loss otherwise.'")
        for loss_node in self.loss_mechs_map:
            # Get output of LossMechanism
            comp_loss = pytorch_rep.nodes_map[loss_node].output
            comp_loss = comp_loss.reshape_as(pytorch_rep.minibatch_loss)
            trial_loss += comp_loss
        return trial_loss

    def _compute_loss_using_standalone_function_and_values_of_output_nodes(self, targets, pytorch_rep, context):
        """Compute loss using values of OUTPUT Nodes as samples
        Loss is computed using a single standalone loss function for all sampe-target pairs
        IMPLEMENTATION NOTE:
            this is legacy code that may be restored for use in the future, though would need to be revised/validated
        """
        # Get value of OUTPUT nodes that are being trained (i.e., for which there are TARGET nodes)
        curr_tensors_for_sample_values = {k:v for k,v in curr_tensors_for_outputs.items()
                                            if k in self.outputs_to_targets_map}

        # Get value of TARGET nodes for current trial
        curr_tensors_for_targets = {}
        for component, target in targets.items():
            if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
                curr_tensors_for_targets[component] = [target[:, :, i, ...] for i in range(target.shape[1])]
            else:
                # It's  a list, of lists, of torch tensors because it is ragged
                num_outputs = len(target[0][0])
                curr_tensors_for_targets[component] = [torch.stack([torch.stack([s[i] for s in b]) for b in target]) for i in range(num_outputs)]

        # Map value of TARGET nodes to trained OUTPUT nodes
        curr_target_tensors_for_sample_values = {}
        for trained_output, target in self.outputs_to_targets_map.items():
            curr_target_tensors_for_sample_values[trained_output] = curr_tensors_for_targets[target]

        # --------- Compute the loss (TARGET-OUTPUT) for each trained OUTPUT node  ---------------------------

        # Calculate and track the loss over the trained OUTPUT nodes:
        #   curr_target_tensors_for_sample_values compared against curr_tensors_for_sample_values
        for component, outputs in curr_tensors_for_sample_values.items():
            trial_loss = 0
            targets = curr_target_tensors_for_sample_values[component]

            num_outputs = outputs.shape[1] if type(outputs) is torch.Tensor else len(outputs[0][0])
            for i in range(num_outputs):
                # loss only accepts 0 or 1d target. reshape assuming pytorch_rep.minibatch_loss dim is correct

                # Get the output, if it's a torch tensor we can slice, if it's a list of list (its ragged) and we
                # need to index
                output = outputs[:, :, i, ...] if type(outputs) is torch.Tensor else torch.stack([torch.stack([s[i] for s in b]) for b in outputs])

                # If the sequence dimension is singleton, it can be dropped
                if len(output.shape) > 1 and output.shape[1] == 1:
                    output = output.squeeze(1)
                    target = torch.atleast_1d(targets[i].squeeze(1))

                comp_loss = self.loss_function(
                    output,
                    target
                )
                comp_loss = comp_loss.reshape_as(pytorch_rep.minibatch_loss)
                trial_loss += comp_loss
            pytorch_rep.minibatch_loss += trial_loss
        pytorch_rep.minibatch_loss_count += 1

        # This may not be needed since it is now taken care of in PytorchCompositionWrapper.forward()
        # --------- Comput the values of output of trained nodes and all nodes  ---------------------------------------

        # IMPLEMENTATION NOTE: Need values in order corresponding to output_CIM Ports.
        # Get output Nodes, their out_ports and corresponding indices
        #     in order of outermost AutodiffComposition's output_CIM Ports
        outputs_idx_port_node_comp = []
        for port in self.output_CIM.input_ports:
            source_info = self.output_CIM._get_source_info_from_output_CIM(port)
            source_ouput_port_idx = source_info[1].output_ports.index(source_info[0])
            # BREADCRUMB: DON'T INCLUDE AS OUTPUT IF IT PROJECTS TO ANOTHER NODE IN AN OUTER COMPOSITION
            outputs_idx_port_node_comp.append(tuple((source_ouput_port_idx, *source_info)))

        # Assign values to trained_output_values and all_output_values
        trained_output_values = []
        all_output_values = []
        for item in outputs_idx_port_node_comp:
            idx, port, node, comp = item
            if comp._trained_comp_nodes_to_pytorch_nodes_map:
                node = comp._trained_comp_nodes_to_pytorch_nodes_map[node]
            outputs = curr_tensors_for_outputs[node]
            if type(outputs) is torch.Tensor:
                output = outputs[:, :, idx, ...]
            else:
                output = torch.stack([torch.stack([s[idx] for s in b]) for b in outputs])

            # If the sequence dimension is singleton, squeeze it away
            if output.shape[1] == 1:
                output = output.squeeze(1)

            output = output.detach().cpu().numpy().copy().tolist()
            # # MODIFIED TEACHER_TARGET OLD:
            # if self.sample_port_to_target_port_map.values():
            # MODIFIED TEACHER_TARGET NEW:
            if self.target_port_to_sample_port_map:
            # MODIFIED TEACHER_TARGET END
                trained_output_values += [output]
            all_output_values += [output]

        return trial_loss


    def clear_losses(self, context=None):
        self.losses = []
        if self.pytorch_representation:
            self.pytorch_representation.retained_losses = []

    def do_gradient_optimization(self, retain_in_pnl_options, context, optimization_num=None):
        """Compute loss and use in call to autodiff_backward() to compute gradients and update PyTorch parameters.
        Update parameters (weights) based on trial(s) executed since last optimization,
        Reinitizalize minibatch_loss and minibatch_loss_count
        """
        pytorch_rep = self.parameters.pytorch_representation._get(context=context)
        minibatch_loss = pytorch_rep.minibatch_loss / pytorch_rep.minibatch_loss_count

        self.autodiff_backward(minibatch_loss, context)

        # # Save loss for current round of optimization
        pytorch_rep.retain_for_psyneulink({LOSSES: minibatch_loss}, retain_in_pnl_options, context)

        # Reset minibatch_loss for next round of optimization
        pytorch_rep.minibatch_loss = torch.zeros(1, device=self.device).double()
        pytorch_rep.minibatch_loss_count = 0

    def autodiff_backward(self, minibatch_loss, context):
        """Calculate gradients and apply to PyTorch model parameters (weights)"""

        if not self.enable_learning:
            return

        pytorch_rep = self.parameters.pytorch_representation._get(context=context)
        optimizer = pytorch_rep.optimizer

        # Gradient updates
        optimizer.zero_grad()
        # Compute and log average loss over all trials since last update
        minibatch_loss.backward(retain_graph=not self.force_no_retain_graph)
        # Update weights and copy to PNL
        optimizer.step()

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "run" in tags:
            return pnlvm.codegen.gen_composition_run(ctx, self, tags=tags)
        else:
            return pnlvm.codegen.gen_autodiffcomp_exec(ctx, self, tags=tags)

    def _get_loss(self, loss_spec):
        if not isinstance(self.loss_spec, (str, Loss)):
            return self.loss_spec
        elif loss_spec == Loss.L1:
            return nn.L1Loss(reduction='sum')
        elif loss_spec == Loss.SSE:
            return nn.MSELoss(reduction='sum')
        elif loss_spec == Loss.MSE:
            return nn.MSELoss(reduction='mean')
        elif loss_spec == Loss.CROSS_ENTROPY:
            if version.parse(torch.version.__version__) >= version.parse('1.12.0'):
                return nn.CrossEntropyLoss()
            # Cross entropy loss is used for multiclass categorization and needs inputs in shape
            # ((# minibatch_size, C), targets) where C is a 1-d vector of probabilities for each potential category
            # and where target is a 1d vector of type long specifying the index to the target category. This
            # formatting is different from most other loss functions available to autodiff compositions,
            # and therefore requires a wrapper function to properly package inputs.
            return lambda x, y: nn.CrossEntropyLoss()(torch.atleast_2d(x), torch.atleast_2d(y.type(x.type())))
        elif loss_spec == Loss.BINARY_CROSS_ENTROPY:
            return nn.BCELoss()
        elif loss_spec == Loss.NLL:
            return nn.NLLLoss(reduction='sum')
        elif loss_spec == Loss.POISSON_NLL:
            return nn.PoissonNLLLoss(reduction='sum')
        elif loss_spec == Loss.KL_DIV:
            return nn.KLDivLoss(reduction='sum')
        else:
            raise AutodiffCompositionError(f"Loss type {loss_spec} not recognized. 'loss_function' argument must be a "
                                           f"Loss enum or function. Currently, the recognized loss types are: "
                                           f"L1 (Mean), SSE (sum squared error), CROSS_ENTROPY, NLL (negative log "
                                           f"likelihood), POISSONNLL (Poisson negative log likelihood, "
                                           f"and KL_DIV (KL divergence.")

    def _get_total_loss(self, num_trials: int=1, context:Context=None):
        return sum(self.parameters.trial_losses._get(context)[-num_trials:]) /num_trials

    def _get_autodiff_inputs_values(self, input_dict: dict):
        """Remove TARGET Nodes, and return dict with values of INPUT Nodes for single trial
        For nested Compositions, replace input to nested Composition with inputs to its INPUT Nodes
        For InuptPorts, replace with owner

        Returns
        ---------
        A dict mapping INPUT Nodes -> input values for a single trial
        """
        autodiff_input_dict = {}
        for node, values in input_dict.items():
            mech = node.owner if isinstance(node, InputPort) else node
            if (mech in self.get_nested_input_nodes_at_all_levels()
                    and mech not in self.get_nodes_by_role(NodeRole.TARGET)):
                # Pass along inputs to all INPUT Nodes except TARGETS
                # (those are handled separately in _get_autodiff_target_node_input_values)
                if torch_available:
                    # Convert to torch tensor of type expected by PytorchCompositionWrapper
                    # values = torch.tensor(values, dtype=self.torch_dtype, device=self.device)
                    values = values.type(self.torch_dtype)
                autodiff_input_dict[node] = values
        return autodiff_input_dict

    def _get_autodiff_target_node_input_values(self, input_dict):
        """Return dict with input values for TARGET Nodes
        Get inputs to TARGET Nodes used for computation of loss in autodiff_forward().
        Use input_dict to get input values for TARGET Nodes that are INPUT Nodes of the AutodiffComposition,
        If a TARGET Node is not an INPUT Node, it is assumed to be an internal target as is ignored,
           as those are assumed to be executed in autodiff_forward()

        Returns
        ---------
        A dict mapping TARGET Nodes -> target values
        """
        target_values = {}
        def get_target_value(target):
            if target in self.get_nodes_by_role(NodeRole.INPUT):
                return input_dict[target]
            if len(target.path_afferents) > 1:
                # TARGET Nodes should only have a single afferent input
                raise AutodiffCompositionError(f"TARGET Node '{target.name}' (for '{self.name}') "
                                               f"cannot have more than one afferent projection.")
            target = target.path_afferents[0].sender.owner
            return get_target_value(target)

        # Get OutputPorts for TARGET Nodes
        for target_port in [t for t in self.target_port_to_sample_port_map
                            if t.owner in self.get_nodes_by_role(NodeRole.TARGET)]:
            target_values[target_port.owner] = get_target_value(target_port.owner)
        return target_values

    def _map_external_target_values_to_target_nodes(self, target_specs: dict, execution_mode)->dict:
        """Map target values to target mechanisms (as needed by learning)

        Returns
        ---------

        `dict`:
            Dict mapping TargetMechanisms -> target values
        """
        target_values_for_target_nodes = {}
        target_mechs = self.get_nodes_by_role(NodeRole.TARGET)


        if execution_mode is not pnlvm.ExecutionMode.PyTorch:
            return super()._map_external_target_values_to_target_nodes(target_specs, execution_mode)

        # Assign target values specified in learn() to TARGET Nodes
        for port, value in target_specs.copy().items():
            # # MODIFIED TEACHER_TARGET OLD:
            # if port in self.sample_port_to_target_port_map:
            #     # Use TARGET Node (target_port owner) for key
            #     target_values_for_target_nodes[self.sample_port_to_target_port_map[port].owner] = value
            # # MODIFIED TEACHER_TARGET NEW:
            # # port to be specified for sample or target; what matters is
            # sample, target = next((item for item in self.sample_port_to_target_port_map.items()
            #                        if port in item), (None,None))
            # MODIFIED TEACHER_TARGET NEWER:
            # port to be specified for sample or target; what matters is
            sample, target = next(((item.sample_port, item.target_port) for item in self.sample_target_pairs
                                   if port in item), (None,None))
            if sample:
                target_values_for_target_nodes[self.sample_port_to_target_port_map[sample].owner] = value
            # MODIFIED TEACHER_TARGET END

        return target_values_for_target_nodes

    def _parse_learn_targets_specs(self, inputs, targets, execution_mode, context, base_context):
        """Override to handle **targets** arguments in construtor and learn() that are specific to AutodiffComposition
        Integrate target specifications from constructor (in self.targets) with those in targets argument of learn():
            handled in override of _aggregate_and_filter_sample_target_specs()
        Deal with nested Compositions
            handled in return from override of this method
        """
        stim_input, num_input_trials = super()._parse_learn_targets_specs(inputs,
                                                                          targets,
                                                                          execution_mode, context, base_context)

        # Replace input to nested Composition with inputs to its INPUT Nodes (to accommodate flattened version)
        if not callable(inputs):
            input_ports_for_INPUT_Nodes = self._get_input_receivers()
            nested_inputs = {}
            stim_input_copy = stim_input.copy()
            for node in stim_input_copy:
                # If node is a nested Composition
                if isinstance(node, Composition):
                    # If owner of input_port is a Node in the nested Composition, replace entry for nested Composition
                    #   in stim_input with entries for the input_ports of its INPUT Nodes
                    for elem, input_port in enumerate([p for p in input_ports_for_INPUT_Nodes if p.owner in node.nodes]):
                        nested_inputs[input_port] = [entry[elem] for entry in stim_input_copy[node]]
                    stim_input.pop(node)
                    stim_input.update(nested_inputs)

        return stim_input, num_input_trials

    # # BREADCRUMB: ??PASS loss_mechs FROM _instantiate_loss_mechs TO AVOID NEED FOR SEARCH OVER self.targets??
    def _parse_constructor_targets_specs(self):
        """Parse sample-target specifications from **targets** of constructor in self._constructor_target_specs
        Standardize format of entries as {sample.output_port: target.output_port or 'TARGET')
        Register samples and targets from LossMechanism specs in loss_mechs_map
        Note: specs have been validated in _validate_targets() for autodiffcomposition.parameters.targets
        """
        spec_as_ouputport = lambda spec : (spec if isinstance(spec, OutputPort)
                                           else (spec.output_port if isinstance(spec, ProcessingMechanism)
                                                 else spec))
        if self.targets:
            # Move self.targets into constructor_target_specs as dict(sample_spec: target_spec)
            for entry in self.targets:
                if isinstance(entry, tuple):
                    self._constructor_target_specs[spec_as_ouputport(entry[0])] = spec_as_ouputport(entry[1])
                elif isinstance(entry, LossMechanism):
                    self._constructor_target_specs[entry] = None
                    self.loss_mechs_map[entry] = (entry.sample, entry.target)

    def _validate_constructor_targets_specs(self):
        """Handle erroneous SAMPLE specs in **targets** argument of constructor
        - Handle redundant specifications and any conflicts among them
            (done in _handle_redundant_sample_target_specs())
        - Check for SAMPLE or TARGET specs NOT in the Composition

        Notes:
        - These are done here and not on Composition, since that does not support specification of SAMPLES
            (there they are assigned automatically as the OUTPUT Nodes of the Composition)
        - The only checks here are for the validity of specifications in the *targets* argument of the constructor
            (at time of construction); compatibilty with specfications in the *targets* argument of learn()
            are handled in _validate_sample_target_specs_from_learn()
        """
        # TEACHER_TARGET BREADCRUMB: MOVE THIS TO globals utlities
        spec_as_mech = lambda spec : (spec.owner if isinstance(spec, OutputPort) else spec)
        def get_inflections(plural):
            inflections = {
                's': 's' if plural else '',
                'a': '' if plural else 'a ',
                'an': '' if plural else 'an ',
                'the': 'the ' if plural else '',
                'are_is': 'are' if plural else 'is',
                'have_has': 'have' if plural else 'has',
                'entry': 'entries' if plural else 'entry',
                'was_were': 'were' if plural else 'was',
            }
            return inflections

        self._handle_redundant_sample_target_specs()

        # Check that all specified Nodes are in the Composition
        nodes_in_comp = self._get_all_nodes()
        not_in_comp = []
        # Get entries in **targets** of constructor for Nodes that are not in the Composition
        # Note: ignore any specified LossMechanisms; they are added in _instantiate_loss_mechanisms()
        for _, sample, _, target, _, _ in self._sample_target_specs:
            if spec_as_mech(sample) not in nodes_in_comp and not isinstance(sample, LossMechanism):
                not_in_comp.append(f"'{sample.full_name}'")
            if spec_as_mech(target) not in nodes_in_comp and not isinstance(target, LossMechanism) and target != TARGET:
                not_in_comp.append(f"'{target.full_name}'")
        if not_in_comp:
            not_in_comp = sorted(set(not_in_comp))
            _ = get_inflections(len(not_in_comp) > 1)
            # X TEST DONE
            raise CompositionError(f"The following specification{_['s']} in the 'targets' argument of the "
                                   f"constructor for '{self.name}' {_['are_is']} not in the Composition or any "
                                   f"nested within it: {', '.join(not_in_comp)}.")

    def _validate_sample_target_specs_from_learn(self, learn_specs, name:str, allow_None_for_target:bool)->dict:
        """Compare learn_specs with constructor specs for SAMPLEs and TARGETs
        Issue error for:
        - missing entries in learn() or ones with a non-numeric value
            for SAMPLEs specified with the keyword 'TARGET' in the constructor
        - *any* specifications for sample-target pairs specified with an internal TARGET Node in the constructor

        Notes:
        - validation of SAMPLES happens in _validate_constructor_targets_specs()
        - for every SAMPLE that has a value = TARGET and source = "autodiff_constructor"
            there should be another entry for that SAMPLE
            that has value == numeric and source = {inputs, inputs[TARGETS or targets}
        - the total number should = the number of SAMPLE Nodes in the Composition:
            if too many: bad specs
            if too few, error (see below)
        """
        # call super() to populate self._sample_target_specs with legal entries from learn() and get illegal ones
        legal_specs, illegal_specs = super()._validate_sample_target_specs_from_learn(learn_specs, name,
                                                                                      allow_None_for_target)

        constructor_specs = self._constructor_target_specs.items()  # specs from **targets** are of constructor
        if not constructor_specs:
            return legal_specs, illegal_specs

        bad_specs = []
        missing_specs = []
        # get SampleTargetSpec from self._sample_target_specs for sample_port
        #    specified in a dict in *targets* of learn()
        _get_learn_spec = lambda spec, spec_list : next((item for item in spec_list
                                                         if (item.sample_port == spec
                                                             and item.source in self._learn_dicts)), None)
        _num_specs = lambda source : len([spec for spec in self._sample_target_specs if spec.source in source])
        # BREADCRUMB: MOVE THIS TO SampleTargets ONCE THAT IS IMPLEMENTED AS A SUBCLASS OF ContentAddressableLisst
        # _get_spec_from_learn = lambda spec : (learn_specs[spec] if spec in learn_specs
        #                                       else (learn_specs[spec.owner] if spec.owner in learn_specs
        #                                             else (learn_specs[spec.name] if spec.name in learn_specs
        #                                                   else (learn_specs[spec.owner.name]
        #                                                         if spec.owner.name in learn_specs
        #                                                         else None))))

        # Iterate over constructor specs to find relevant ones in either illegal_specs or self._sample_target_specs
        for sample, target in constructor_specs:
            # Check if sample is in illegal_specs from learn()
            learn_spec = _get_learn_spec(sample, illegal_specs)
            if learn_spec:
                illegal_specs.remove(learn_spec)
            else:
                # Check if sample was in learn()
                learn_spec = _get_learn_spec(sample, self._sample_target_specs)
            if learn_spec:
                # sample specified in learn()
                learn_sample, learn_target, learn_value = (learn_spec.sample_port,
                                                           learn_spec.target_spec,
                                                           learn_spec.target_value)
                if target == TARGET:
                    # sample should be specified in learn() with numeric value
                    # # MODIFIED TEACHER_TARGET OLD:
                    # if is_numeric(learn_value):
                    #     # target in constructor is specified as TARGET, and spec in learn() is correctly numeric
                    #     # BREADCRUMB: IS THE FOLLOWING OK SINCE IT IS PASSED IN?
                    #     # legal_specs.update({learn_spec: learn_value})
                    #     pass
                    # else:
                    # MODIFIED TEACHER_TARGET NEW:
                    if not is_numeric(learn_value):
                    # MODIFIED TEACHER_TARGET END
                        # X TEST DONE: EXPECTED NUMERIC BUT GOT NON-NUMERIC
                        # expected numeric spec for target in learn(), but got non-numeric
                        bad_specs.append((learn_spec, learn_target,
                                          f"'TARGET' is assigned as its value in the 'targets' argument of the "
                                          f"constructor, specifying that its actual value is provided as an input "
                                          f"in learn(), so the entry for it in the 'targets' argument of learn() "
                                          f"must be a numeric value (rather than '{learn_spec.target_value})'"))
                else:
                    # sample should NOT be specified in learn()
                    # Got unexpected specification for target in learn() (speccified as Node in constructor)
                    # X TEST DONE: EXPECTED NO SPECIFICATION BUT GOT ONE
                    bad_specs.append((learn_spec, learn_target,
                                      f"a Node ('{learn_spec.target_spec.full_name}') that provides its target value "
                                      f"is specified in the 'targets' argument of the constructor, so there should "
                                      f"be no specification for the SAMPLE in learn()"))
            else:
                # sample NOT specified in learn()
                if target == TARGET:
                    # X TEST DONE: EXPECTED NUMERIC BUT GOT NOTHING
                    # Missing numeric specification for target in learn() (specified as TARGET in constructor)
                    bad_specs.append((learn_spec, learn_target,
                                      f"the sample is assigned 'TARGET' as its value in the 'targets' argument of the "
                                      f"constructor, so it should also be specified in the 'targets argument of the "
                                      "learn method, and assigned a numeric array (i.e., the value used for training "
                                      "on each trial."))

        # MODIFIED TEACHER_TARGET NEWER:
        # Add any illegal specs passed in to bad_specs for reporting in error message
        for spec in illegal_specs:
            illegal_spec = (
                f"'{spec.target_spec.full_name}'"
                if isinstance(spec.target_spec, (OutputPort, ProcessingMechanism_Base))
                else spec.target_spec
            )
            bad_specs.append((spec, illegal_spec, f"does not correspond to any sample specified "
                                                  f"in the constructor"))
        # MODIFIED TEACHER_TARGET END

        if bad_specs:
            # BREADCRUMB: MOVE THIS TO SamplePairs class ONCE THAT IS IMPLEMENTED
            def get_inflections(plural):
                inflections = {
                    's': 's' if plural else '',
                    'not_s': '' if plural else 's',
                    'a': '' if plural else 'a ',
                    'an': '' if plural else 'an ',
                    'the': 'the ' if plural else '',
                    'are_is': 'are' if plural else 'is',
                    'have_has': 'have' if plural else 'has',
                    'entry': 'entries' if plural else 'entry',
                    'was_were': 'were' if plural else 'was',
                }
                return inflections
            _ = get_inflections(len(bad_specs)>1)
            all_bad_specs_str = []
            sources = []
            for bad_spec in bad_specs:
                sources.append(bad_spec[0].source)
                if bad_spec[0].sample_spec:
                    all_bad_specs_str.append(f"for SAMPLE '{bad_spec[0].sample_spec.full_name}': {bad_spec[2]}")
                else:
                    all_bad_specs_str.append(f"{bad_spec[1]}: {bad_spec[2]}")

            sources = sorted(set(sources))
            if len(sources) == 1:
                s = ''
                source_str = f"'{sources[0]}'"
            else:
                s = 's'
                source_str = (' and '.join([f"'{str}'" for str in sources]) if len(sources)==2
                              else f"'{sources[0]}', '{sources[1]}' and '{sources[2]}'")
            source_str = f"{source_str} argument{s}"

            raise AutodiffCompositionError(f"The learn() method of '{self.name}' can't be executed because "
                                           f"the following specification{_['s']} in its {source_str} "
                                           f"conflict{_['not_s']} with one{_['s']} in the 'targets' argument "
                                           f"of its consructor: {'; '.join(all_bad_specs_str)}.")

        return legal_specs, illegal_specs

    def _get_redundant_sample_target_specs(self):
        """Override to allow specification of TARGET in constructor and required numeric value in learn()"""
        all_redundant_specs = super()._get_redundant_sample_target_specs()
        for spec in all_redundant_specs.copy():
            # Get target_spec and target_value for each of the redundant specs
            redundant_specs = [s for s in self._sample_target_specs if s.sample_port is spec]
            if len(redundant_specs) == 2:
                constructor_spec, learn_spec = redundant_specs
                if (constructor_spec.target_spec == TARGET
                        and constructor_spec.source == CONSTRUCTOR_TARGETS
                        and learn_spec.source != CONSTRUCTOR_TARGETS
                        and is_numeric(learn_spec.target_value)):
                    all_redundant_specs.remove(spec)
        return all_redundant_specs

    def _handle_conflicting_sample_target_specs(self, samples_with_mismatching_specs:list):
        """Override to handle conflict between sample specs and/or values from constructor and learn()
        Handle conflicts between different target values specified for:
             same SAMPLE Nodes specified in constructor using different references (e.g., mech vs. mech.output_port)
             SAMPLE in constructor vs. learn() (e.g., Node in constructor vs. numeric value in learn())
        """
        # BREADCRUMB: FROM Composition --- NEED TO ADAPT TO DEAL WITH CONFLICTS BEETWEEN VALUES FROM LEARN (NUMBERS)
        #              AND VALUES FROM CONSTRUCTOR (NODES OR "TARGET")

        # Error for redundant specs with different values
        # -----------------------------------------------
        # prepare strings for warning message
        all_samples_str = []
        for sample, values in samples_with_mismatching_specs.items():
            specs_and_sources = [f"{s.target_spec if isinstance( s.target_spec, str) else s.target_spec.full_name} in "
                                 f"'{s.source}'" for s in self._sample_target_specs
                                 for value in values if s.sample_port is sample and s.target_spec is value]
            sources_str = ', '.join(specs_and_sources)
            all_samples_str.append(f"'{sample.full_name}': {sources_str}")
        full_str = '; '.join(all_samples_str)

        if full_str:
            # BREADCRUMB: INTERGRATE THIS WITH inflections IN _validate_constructor_targets_specs
            many_conflicts = len(all_samples_str) > 1
            many_outputs = len(self.get_nodes_by_role(NodeRole.OUTPUT))
            s = 's' if many_conflicts else ''
            multiple = ' multiple' if many_conflicts else ""
            one_of = 'one of ' if (many_outputs and not many_conflicts) else ''
            node_s = 's' if many_outputs else ''
            # BREADCRUMB: HANDLE OUTPUT IN OVERRIDE IN AutodiffComposition
            # sample_nodes = 'SAMPLE' if self._constructor_has_target_specs else 'OUTPUT'
            # X TEST DONE
            raise CompositionError(f"The learn() method of '{self.name}' can't be executed because there are{multiple} "
                                   f"conflicting specifications for the target value{s} assigned to {one_of}its "
                                   f"SAMPLE Node{node_s}: {full_str}.")

    def _check_nested_target_mechs(self):
        pass

    def _identify_output_nodes(self, context)->list:
        """Recursively call all nested AutodiffCompositions to assign TARGET nodes for learning"""
        # Default is to use OUTPUT
        output_nodes = set(node for node in self.get_nodes_by_role(NodeRole.OUTPUT)
                           if not isinstance(node, Composition))
        for node in self.nodes:
            if isinstance(node, AutodiffComposition):
                output_nodes = output_nodes.union(node._identify_output_nodes(context))
        return output_nodes

    def _get_valid_weights_shape(self, projection):
        pnl_wt_matrix = projection.defaults.matrix
        if not isinstance(pnl_wt_matrix, np.ndarray):
            assert is_matrix_keyword(pnl_wt_matrix)
            pnl_wt_matrix = projection._get_matrix_from_keyword(pnl_wt_matrix)
        return pnl_wt_matrix.shape

    @handle_external_context()
    def set_weights(self, pnl_proj, weights:Union[list, np.ndarray], context=None):
        """Set weights for specified Projection."""
        valid_shape = self._get_valid_weights_shape(pnl_proj)
        assert weights.shape == valid_shape, \
            (f"PROGRAM ERROR: Shape of weights in 'weights' arg of '{self.name}.set_weights' "
             f"Specified weights do not match required shape ({valid_shape}).)")
        pnl_proj.parameters.matrix._set(weights, context)
        pnl_proj.parameter_ports['matrix'].parameters.value._set(weights, context)

    @handle_external_context(fallback_default=True)
    def learn(self,
              *args,
              execute_in_additional_optimizations: Optional[dict] = None,
              synch_projection_matrices_with_torch: SynchRetainArg = NotImplemented,
              synch_node_variables_with_torch: SynchRetainArg = NotImplemented,
              synch_node_values_with_torch: SynchRetainArg = NotImplemented,
              synch_results_with_torch: SynchRetainArg = NotImplemented,
              retain_torch_sample_values: SynchRetainArg = NotImplemented,
              retain_torch_targets: SynchRetainArg = NotImplemented,
              retain_torch_losses: SynchRetainArg = NotImplemented,
              context: Context = None,
              base_context: Context = Context(execution_id=None),
              skip_initialization: bool = False,
              **kwargs
              ) -> list:
        """Override to handle synch and retain args; see `Composition.run` for additional arguments and details.

        .. _technical_note::
           defaults for synch and retain args are set to NotImplemented, so that the user can specify None if they
           want to locally override the default values for the AutodiffComposition (see docstrings for run() and
           parse_synch_and_retain_args() for additonal details).

        Arguments
        ---------

        learning_rate : float, int, bool or dict : default 0.001
            specifies the learning rate(s) passed to the optimizer, that overrides any learning_rate specifications
            made in AutodiffComposition constructor and/or individual MappingProjections. If a value is specified,
            it overrides the default learning rate for the Composition, and is used as the default learning rate for
            all MappingProjections in the Composition (and any nested within it) that do not have a specific
            learning_rate specified in their constructor.  A dict can be used to specify
            `MappingProjection`\\-specific learning_rate(s); if it contains a *DEFAULT_LEARNING_RATE* entry,
            that is used in the same was as specifing numeric value; if the dict does not contain a
            *DEFAULT_LEARNING_RATE* entry, then the default indicated above is used for all MappingProjections
            in the Composition, and MappingProjections in any nested Compositions use their default learning_rate
            (see `AutodiffComposition_Learning_Rate` and `Composition_Learning_Rate` for additional details).

        execute_in_additional_optimizations : dict{`Node <Composition_Nodes>`:[(Parameter, value)]} (default None)
            specifies which `Nodes <Composition_Nodes>` of the AutodiffComposition should be included in the forward
            pass for any additional optimization steps after the first; this overrides any specifications made in the
            **execute_in_additional_optimizations** argument of the AutodiffComposition's constructor (see
            `AutodiffComposition_Optimization_Steps` for fuller explanation and details of specification).

        synch_projection_matrices_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_projection_matrices_with_torch
            <AutodiffComposition.synch_projection_matrices_with_torch>` for additional details.

        synch_node_variables_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_node_variables_with_torch
            <AutodiffComposition.synch_node_variables_with_torch>` for additional details.

        synch_node_values_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_node_values_with_torch
            <AutodiffComposition.synch_node_values_with_torch>` for additional details.

        synch_results_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_results_with_torch
            <AutodiffComposition.synch_results_with_torch>` for additional details.

        retain_torch_sample_values : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `retain_torch_sample_values
            <AutodiffComposition.retain_torch_sample_values>` for additional details.

        retain_torch_targets : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `retain_torch_targets
            <AutodiffComposition.retain_torch_targets>` for additional details.

        retain_torch_losses : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `retain_torch_losses
            <AutodiffComposition.retain_torch_losses>` for additional details.
        """
        # NOTE: do not call _initialize_from_context here -
        # infer_backpropagation_learning_pathways call below can change
        # the structure of the Composition and its CIMs and this will
        # result in them having old values. Stateful Parameter get may
        # not have a value before call to super().learn

        execution_phase_at_entry = context.execution_phase
        context.execution_phase = ContextFlags.PREPARING

        execution_mode = self._get_execution_mode(kwargs.pop('execution_mode', None))
        context.execution_phase = execution_phase_at_entry

        learning_rate = kwargs.get(LEARNING_RATE, None)
        # Deal with deprecated arg (can't use deprecation_warning() since that is for constructors)
        if OPTIMIZER_PARAMS in kwargs:
            default_learning_rate = kwargs.pop(LEARNING_RATE, None)
            learning_rate = deprecation_warning(self, kwargs,
                                                deprecated_args={OPTIMIZER_PARAMS:LEARNING_RATE},
                                                method="learn() method",
                                                additional_msg=" Other torch.nn.optimizer parameters are not "
                                                               "currently supported, but will be in a future version.")

        runtime_optimizer_params = OptimizerParams(learning_rate=copy.copy(learning_rate))

        if OPTIMIZER_PARAMS in kwargs:
            kwargs.update(learning_rate)
            # Move learning_rate spec into optimizer_params dict
            if default_learning_rate is not None:
                if kwargs[LEARNING_RATE]:
                    kwargs[LEARNING_RATE].update({DEFAULT_LEARNING_RATE: default_learning_rate})
                else:
                    kwargs[LEARNING_RATE] = {DEFAULT_LEARNING_RATE: default_learning_rate}

        if LEARNING_RATE in kwargs and isinstance(kwargs[LEARNING_RATE], dict):
            # If learning_rate is a dict:
            # - move it to optimizer_params;
            # - if it contains DEFAULT_LEARNING_RATE entry, assign that as learning_rate
            kwargs[OPTIMIZER_PARAMS] = kwargs[LEARNING_RATE]
            kwargs[LEARNING_RATE] = kwargs[OPTIMIZER_PARAMS].pop(DEFAULT_LEARNING_RATE, None)

        any_nested_comps = [node for node in self.nodes if isinstance(node, Composition)]
        if any_nested_comps:
            # Can't learn in Python mode if any nested Compositions
            if execution_mode is not pnlvm.ExecutionMode.PyTorch:
                nested_comp_names = [f"'{comp.name}'" for comp in any_nested_comps]
                raise AutodiffCompositionError(f"Unable to execute learning in {pnlvm.ExecutionMode.Python.name} mode "
                                               f"for '{self.name}' because it contains one or more nested "
                                               f"Compositions: {' ,'.join(nested_comp_names)}.")

            # Can't learn if any nested comps that are not AutodiffCompositions
            nested_comps = [f"'{comp.name}'" for comp in any_nested_comps if not isinstance(comp, AutodiffComposition)]
            if nested_comps:
                raise AutodiffCompositionError(f"Unable execute learning for '{self.name}' "
                                               f"because it contains nested Composition(s) "
                                               f"that are not AutodiffCompositions: {' ,'.join(nested_comps)}.")

        if self._built_pathways is False:
            # TEACHER_TARGET BREADCRUMB: ADD TEST HERE FOR LEARNABLE PATHWAYS AND WARN IF NONE
            if not self._has_learnable_pathways:
                raise AutodiffCompositionError(f"'{self.name}' does not have any learnable pathways, "
                                               f"therefore its learn() method cannot be executed.")
            self.infer_backpropagation_learning_pathways(execution_mode, context=context, base_context=base_context)
            self._built_pathways = True


        synch_with_pnl_options, retain_in_pnl_options = self.parse_synch_and_retain_args(
            context,
            synch_projection_matrices_with_torch=synch_projection_matrices_with_torch,
            synch_node_variables_with_torch=synch_node_variables_with_torch,
            synch_node_values_with_torch=synch_node_values_with_torch,
            synch_results_with_torch=synch_results_with_torch,
            retain_torch_sample_values=retain_torch_sample_values,
            retain_torch_targets=retain_torch_targets,
            retain_torch_losses=retain_torch_losses,
        )

        if execution_mode == pnlvm.ExecutionMode.PyTorch and not torch_available:
            raise AutodiffCompositionError(f"'{self.name}.learn()' has been called with ExecutionMode.Pytorch, "
                                           f"but Pytorch module ('torch') is not installed. "
                                           f"Please install it with `pip install torch` or `pip3 install torch`")

        return super().learn(*args,
                             synch_with_pnl_options=synch_with_pnl_options,
                             retain_in_pnl_options=retain_in_pnl_options,
                             execution_mode=execution_mode,
                             context=context,
                             base_context=base_context,
                             skip_initialization=skip_initialization,
                             # TODO: rename/replace this with just optimizer_params
                             runtime_optimizer_params=runtime_optimizer_params,
                             **kwargs)

    def parse_synch_and_retain_args(
        self,
        context: Context,
        synch_projection_matrices_with_torch: SynchRetainArg = NotImplemented,
        synch_node_variables_with_torch: SynchRetainArg = NotImplemented,
        synch_node_values_with_torch: SynchRetainArg = NotImplemented,
        synch_results_with_torch: SynchRetainArg = NotImplemented,
        retain_torch_sample_values: SynchRetainArg = NotImplemented,
        retain_torch_targets: SynchRetainArg = NotImplemented,
        retain_torch_losses: SynchRetainArg = NotImplemented,
    ) -> Tuple[Dict, Dict]:
        return self._parse_synch_and_retain_args(
            context,
            synch_projection_matrices_with_torch=synch_projection_matrices_with_torch,
            synch_node_variables_with_torch=synch_node_variables_with_torch,
            synch_node_values_with_torch=synch_node_values_with_torch,
            synch_results_with_torch=synch_results_with_torch,
            retain_torch_sample_values=retain_torch_sample_values,
            retain_torch_targets=retain_torch_targets,
            retain_torch_losses=retain_torch_losses,
        )

    def _parse_synch_and_retain_args(
        self, context: Context, **kwargs
    ) -> Tuple[Dict, Dict]:
        # Package options for synching and tracking into dictionaries as arguments to learning and exec methods
        def _get_option_val(arg):
            arg_param = getattr(self.parameters, arg)
            val = kwargs.get(arg, NotImplemented)
            try:
                val = LearningScale(val)
            except ValueError:
                # val could be None, which is an acceptable SynchRetainArg
                pass
            if val is NotImplemented:
                val = arg_param._get(context, fallback_value=NotImplemented)
            if val is NotImplemented:
                val = arg_param.default_value
            return val

        # consider making these Parameter aliases
        synch_with_pnl_options = {
            MATRIX_WEIGHTS: "synch_projection_matrices_with_torch",
            NODE_VARIABLES: "synch_node_variables_with_torch",
            NODE_VALUES: "synch_node_values_with_torch",
            RESULTS: "synch_results_with_torch",
        }
        retain_in_pnl_options = {
            SAMPLE_VALUES: "retain_torch_sample_values",
            TARGETS: "retain_torch_targets",
            LOSSES: "retain_torch_losses",
        }
        for result_name, arg in synch_with_pnl_options.items():
            synch_with_pnl_options[result_name] = _get_option_val(arg)
        for result_name, arg in retain_in_pnl_options.items():
            retain_in_pnl_options[result_name] = _get_option_val(arg)

        if self.minibatch_size > 1:
            args_str = []
            if retain_in_pnl_options[SAMPLE_VALUES] in {LearningScale.OPTIMIZATION_STEP, LearningScale.TRIAL}:
                args_str.append('retain_torch_sample_values')
            if retain_in_pnl_options[LOSSES] in {LearningScale.OPTIMIZATION_STEP, LearningScale.TRIAL}:
                args_str.append('retain_torch_losses')
            if retain_in_pnl_options[TARGETS] in {LearningScale.OPTIMIZATION_STEP, LearningScale.TRIAL}:
                args_str.append('retain_torch_targets')
            if args_str:
                arg_args = 'args' if len(args_str) == 1 else 'arg'
                is_are = 'is' if len(args_str) == 1 else 'are'
                raise AutodiffCompositionError(f"The {' ,'.join(args_str)} {arg_args} in the learn() method for "
                                               f"'{self.name}' {is_are} specifed as 'OPTIMIZATION' or 'TRIAL', but "
                                               f"'minibatch_size` ({self.minibatch_size}) != 1, so "
                                               f"{', '.join([arg.split('_')[-1] for arg in args_str])} "
                                               f"will be updated only at the end of a minibatch; "
                                               f"use 'MINIBATCH' for the {arg_args} to avoid this warning.")

        return synch_with_pnl_options, retain_in_pnl_options

    def _get_execution_mode(self, execution_mode):
        """Parse execution_mode argument and return a valid execution mode for the learn() method
        Can be overridden by subclasses to change the permitted and/or default execution mode for learning
        """
        if execution_mode is None:
            if self._warned_about_default_execution_mode is False:
                warnings.warn(f"The execution_mode argument was not specified in the learn() method of '{self.name}'; "
                              f"ExecutionMode.PyTorch will be used by default.")
                self._warned_about_default_execution_mode = True
            execution_mode = pnlvm.ExecutionMode.PyTorch

        return execution_mode

    @handle_external_context(fallback_default=True)
    def execute(self,
                inputs=None,
                num_trials=None,
                minibatch_size=1,
                optimizations_per_minibatch=1,
                optimization_num=None,
                do_logging=False,
                scheduler=None,
                termination_processing=None,
                call_before_minibatch=None,
                call_after_minibatch=None,
                call_before_time_step=None,
                call_before_pass=None,
                call_after_time_step=None,
                call_after_pass=None,
                reset_stateful_functions_to=None,
                context=None,
                base_context=Context(execution_id=None),
                clamp_input=SOFT_CLAMP,
                targets=None,
                optimizer_params:dict=None,
                runtime_params=None,
                execution_mode:pnlvm.ExecutionMode = pnlvm.ExecutionMode.PyTorch,
                skip_initialization=False,
                synch_with_pnl_options:Optional[Mapping]=None,
                retain_in_pnl_options:Optional[Mapping]=None,
                report_output:ReportOutput=ReportOutput.OFF,
                report_params:ReportOutput=ReportParams.OFF,
                report_progress:ReportProgress=ReportProgress.OFF,
                report_simulations:ReportSimulations=ReportSimulations.OFF,
                report_to_devices:ReportDevices=None,
                report=None,
                report_num=None,
                )->np.ndarray:
        """Override to execute autodiff_forward() in learning mode if execute_mode is not Python"""

        if (self._is_learning(context) and execution_mode is not pnlvm.ExecutionMode.PyTorch and
                any([isinstance(node, Composition) for node in self.nodes])):
            raise CompositionError(f"Must use execution_mode=ExecutionMode.PyTorch for learning "
                                   f"that includes nested AutodiffComposition(s).")

        if execution_mode is not pnlvm.ExecutionMode.Python:

            if scheduler is None:
                scheduler = self.scheduler

            # TBI: How are we supposed to use base_context and statefulness here?
            if ContextFlags.LEARNING_MODE in context.runmode:
                autodiff_inputs = self._get_autodiff_inputs_values(inputs)
                autodiff_targets = self._get_autodiff_target_node_input_values(inputs)

                # Begin reporting of learning TRIAL:
                report(self,
                       LEARN_REPORT,
                       # EXECUTE_REPORT,
                       report_num=report_num,
                       scheduler=scheduler,
                       content='trial_start',
                       context=context)

                output_values = self.autodiff_forward(inputs=autodiff_inputs,
                                                      targets=autodiff_targets,
                                                      optimization_num=optimization_num,
                                                      synch_with_pnl_options=synch_with_pnl_options,
                                                      retain_in_pnl_options=retain_in_pnl_options,
                                                      execution_mode=execution_mode,
                                                      scheduler=scheduler,
                                                      context=context)
                execution_phase = context.execution_phase
                context.execution_phase = ContextFlags.PROCESSING
                context.execution_phase = execution_phase

                # Complete TRIAL Panel for output report, and report progress
                report(self,
                       # [LEARN_REPORT],
                       [EXECUTE_REPORT, PROGRESS_REPORT],
                       report_num=report_num,
                       scheduler=scheduler,
                       content='trial_end',
                       context=context)

                scheduler.get_clock(context)._increment_time(TimeScale.TRIAL)

                self.most_recent_context = context
                return output_values


        # Call Composition execute in Python mode
        return super(AutodiffComposition, self).execute(inputs=inputs,
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
                                                        optimizer_params=optimizer_params,
                                                        runtime_params=runtime_params,
                                                        execution_mode=execution_mode,
                                                        report=report,
                                                        report_num=report_num
                                                        )

    @handle_external_context(fallback_default=True)
    def run(self, *args,
            execution_mode: pnlvm.ExecutionMode = pnlvm.ExecutionMode.Python,
            synch_projection_matrices_with_torch: SynchRetainArg = NotImplemented,
            synch_node_variables_with_torch: SynchRetainArg = NotImplemented,
            synch_node_values_with_torch: SynchRetainArg = NotImplemented,
            synch_results_with_torch: SynchRetainArg = NotImplemented,
            retain_torch_sample_values: SynchRetainArg = NotImplemented,
            retain_torch_targets: SynchRetainArg = NotImplemented,
            retain_torch_losses: SynchRetainArg = NotImplemented,
            batched_results:bool=False,
            context: Union[Context, Hashable] = None,
            base_context: Context = Context(execution_id=None),
            **kwargs):
        """Override to handle synch and retain args if called directly from run() rather than learn()
        Note: defaults for synch and retain args are NotImplemented, so that the user can specify None if they want
              to locally override the default values for the AutodiffComposition (see parse_synch_and_retain_args()
              for details). This is distinct from the user assigning the Parameter default_values(s), which is done
              in the AutodiffComposition constructor and handled by the Parameter._specify_none attribute.
        """
        # NOTE: like in .learn, do not call _initialize_from_context
        # here. correct shapes for CIMs are determined in .run before
        # _initialize_from_context is called there.

        # Store whether we need to return results list with a batch dimension, or flatten it
        self.batched_results = batched_results

        if not (SYNCH_WITH_PNL_OPTIONS in kwargs and RETAIN_IN_PNL_OPTIONS in kwargs):
            # No synch_with_pnl_options and retain_in_pnl_options dicts:
            # - so must have been called from run directly rather than learn
            # - therefore, must validate, parse and package options into those dicts
            if synch_results_with_torch is NotImplemented:
                # IMPLEMENTATION NOTE:
                #     If synch_results_with_torch is not specified by the user in call from run(), set it to
                #     MINIBATCH (rather than RUN, which is the default_value for calls from AutodiffComposition);
                #     this is required for calling _update_results() from Composition.run(), which does not itself
                #     know about synch and retain options, and the expected default behavior of which is to update
                #     results on every try in a call to run().
                synch_results_with_torch = LearningScale.MINIBATCH
            synch_with_pnl_options, retain_in_pnl_options = self.parse_synch_and_retain_args(
                context,
                synch_projection_matrices_with_torch=synch_projection_matrices_with_torch,
                synch_node_variables_with_torch=synch_node_variables_with_torch,
                synch_node_values_with_torch=synch_node_values_with_torch,
                synch_results_with_torch=synch_results_with_torch,
                retain_torch_sample_values=retain_torch_sample_values,
                retain_torch_targets=retain_torch_targets,
                retain_torch_losses=retain_torch_losses,
            )
            kwargs[SYNCH_WITH_PNL_OPTIONS] = synch_with_pnl_options
            kwargs[RETAIN_IN_PNL_OPTIONS] = retain_in_pnl_options

        # In LEARNING_MODE, so check that at least one enable_learning is True (potentially in nested Comp)
        if ContextFlags.LEARNING_MODE in context.runmode and not (self._is_learning(context) or
                                                                  any(comp._is_learning(context)
                                                                      for comp in self._get_nested_compositions())):
            raise AutodiffCompositionError(f"The learn() method of '{self.name}' was called, but its "
                                           f"'enable_learning' Parameter (and the ones for any Compositions "
                                           f"nested within) it are set to 'False'. Either set at least one to "
                                           f"'True', or use {self.name}.run().")

        if execution_mode != pnlvm.ExecutionMode.Python and ContextFlags.LEARNING_MODE in context.runmode:
            self._assign_execution_ids(context)
            context.composition = self
            context.source = ContextFlags.COMPOSITION

            if execution_mode is pnlvm.ExecutionMode.PyTorch and not torch_available:
                raise AutodiffCompositionError(f"'{self.name}.learn()' has been called with ExecutionMode.Pytorch, "
                                               f"but Pytorch module ('torch') is not installed. "
                                               f"Please install it with `pip install torch` or `pip3 install torch`")
            self._build_pytorch_representation(optimizer_params=kwargs.get('optimizer_params', None),
                                               context=context,
                                               base_context=Context(execution_id=None))

        # Run AutodiffComposition
        results = super(AutodiffComposition, self).run(*args, execution_mode=execution_mode, context=context, **kwargs)

        if execution_mode == pnlvm.ExecutionMode.PyTorch:
            # Synchronize specified outcomes at end of run
            pytorch_rep = self.parameters.pytorch_representation.get(context)
            if pytorch_rep:
                # # MODIFIED TEACHER_TARGET OLD:
                # pytorch_rep.synch_with_psyneulink(kwargs[SYNCH_WITH_PNL_OPTIONS], LearningScale.RUN, context)
                # MODIFIED TEACHER_TARGET NEW:
                # Synchronize with PsyNeuLink at end of run if in learning mode (i.e., called from learn()),
                if context.runmode == ContextFlags.LEARNING_MODE:
                    pytorch_rep.synch_with_psyneulink(kwargs[SYNCH_WITH_PNL_OPTIONS], LearningScale.RUN, context)
                else:
                # But not if called directly from run(), since that is always executed in Python
                # irrespective of execution_mode, so pytorch_representation will not be updated,
                # and may not have executed at all.
                    warnings.warn(f"Although the run() method of '{self.name}' was called with "
                                  f"'execution_mode=ExecutionMode.PyTorch' it will be run in Python mode; an "
                                  f"AutodiffComposition is only executed in PyTorch mode when its learn() is called.")
                # MODIFIED TEACHER_TARGET END

        return results

    def _update_results(self, results, trial_output, execution_mode, synch_with_pnl_options, context):
        """Track results at specified frequency during learning"""
        if execution_mode is pnlvm.ExecutionMode.PyTorch:

            # Check if the trial_output is atleast 3D
            is_output_3d = trial_output.ndim >= 3 or (trial_output.ndim == 2 and len(trial_output) > 0 and
                                                      isinstance(trial_output[0, 0], (np.ndarray, list)))

            if (RESULTS in synch_with_pnl_options
                    and synch_with_pnl_options[RESULTS] in {LearningScale.TRIAL, LearningScale.MINIBATCH}):
                # Use Composition's own _update_results method since no savings when done trial-by-trial
                if not self.batched_results and is_output_3d:
                    for out in trial_output:
                        super()._update_results(results, out, execution_mode, synch_with_pnl_options, context)
                else:
                    super()._update_results(results, trial_output, execution_mode, synch_with_pnl_options, context)

            elif (RESULTS in synch_with_pnl_options
                  and synch_with_pnl_options[RESULTS] == LearningScale.RUN):
                # Use pytorch_reps method to keep a local list of results that are copied to autodiff.results after run
                pytorch_rep = self.parameters.pytorch_representation._get(context)
                if not self.batched_results and is_output_3d:
                    for out in trial_output:
                        pytorch_rep.retain_results(out)
                else:
                    pytorch_rep.retain_results(trial_output)
        else:
            super()._update_results(results, trial_output, execution_mode, synch_with_pnl_options, context)

    @handle_external_context(fallback_most_recent=True)
    def save(self, path:PosixPath=None, directory:str=None, filename:str=None, context=None):
        """Saves all weight matrices for all MappingProjections in the AutodiffComposition

        Arguments
        ---------
        path: Path, PosixPath or str : default None
            path specification; must be a legal path specification in the filesystem.
        directory: str : default ``current working directory``
            directory where `matrices <MappingProjection.matrix>` for all MappingProjections
            in the AutodiffComposition are saved.
        filename: str : default ``<name of AutodiffComposition>_matrix_wts.pnl``
            filename in which `matrices <MappingProjection.matrix>` for all MappingProjections
            in the AutodiffComposition are saved.
        .. note::
           Matrices are saved in
           `PyTorch state_dict <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_ format.

        Return
        ------
        Path

        """
        error_msg = f" (for saving weight matrices for '{self.name}') is not a legal path."

        if path:
            try:
                path = Path(path)
            except:
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        else:
            try:
                if directory:
                    path = Path(directory)
                else:
                    path = Path(os.getcwd())
                if filename:
                    path = Path(os.path.join(path, filename))
                else:
                    path = Path(os.path.join(path, f'{self.name}_matrix_wts.pnl'))
            except IsADirectoryError:
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        proj_state = {
            p.name: p.parameters.matrix.get(context=context)
            # p.name: p.matrix.base
            for p in self.projections
            if not (isinstance(p, ModulatoryProjection_Base)
                    or isinstance(p.sender.owner, CompositionInterfaceMechanism)
                    or isinstance(p.receiver.owner, CompositionInterfaceMechanism)
                    or isinstance(p.sender.owner, ModulatoryMechanism_Base)
                    or isinstance(p.receiver.owner, ModulatoryMechanism_Base)
                    or p.sender.owner in self.get_nodes_by_role(NodeRole.LEARNING)
                    or p.receiver.owner in self.get_nodes_by_role(NodeRole.LEARNING)
                )}
        try:
            torch.save(proj_state, path)
        except IsADirectoryError:
            raise AutodiffCompositionError(f"'{path}'{error_msg}")

        self.last_saved_weights = path

        return path

    @handle_external_context(fallback_most_recent=True)
    def load(self, path:PosixPath=None, directory:str=None, filename:str=None, context=None, weights_only:bool=False):
        """Loads all weight matrices for all MappingProjections in the AutodiffComposition from file
        Arguments
        ---------
        path: Path : default None
            Path for file in which `MappingProjection` `matrices <MappingProjection.matrix>` are stored.
            This must be a legal PosixPath object; if it is specified **directory** and **filename** are ignored.
        directory: str : default ``current working directory``
            directory where `MappingProjection` `matrices <MappingProjection.matrix>` are stored.
        filename: str : default ``<name of AutodiffComposition>_matrix_wts.pnl``
            name of file in which `MappingProjection` `matrices <MappingProjection.matrix>` are stored.
        .. note::
           Matrices must be stored in
           `PyTorch state_dict <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_ format.
        """
        error_msg = f" (for loading weight matrices for '{self.name}') is not a legal path."
        if path:
            if not isinstance(path,Path):
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        else:
            try:
                if directory:
                    path = Path(directory)
                else:
                    path = Path(os.getcwd())
                if filename:
                    path = Path(os.path.join(path, filename))
                else:
                    path = Path(os.path.join(path , f'{self.name}_matrix_wts.pnl'))
            except IsADirectoryError:
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        try:
            state = torch.load(path, weights_only=weights_only)
        except FileNotFoundError:
            raise AutodiffCompositionError(f"'{path}'{error_msg}")

        self.last_loaded_weights = path

        for projection in [p for p in self.projections
                           if not (isinstance(p, ModulatoryProjection_Base)
                                   or isinstance(p.sender.owner, CompositionInterfaceMechanism)
                                   or isinstance(p.receiver.owner, CompositionInterfaceMechanism)
                                   or isinstance(p.sender.owner, ModulatoryMechanism_Base)
                                   or isinstance(p.receiver.owner, ModulatoryMechanism_Base)
                                   or p.sender.owner in self.get_nodes_by_role(NodeRole.LEARNING)
                                   or p.receiver.owner in self.get_nodes_by_role(NodeRole.LEARNING)
            )]:
            matrix = state[projection.name]
            if np.array(matrix).shape != projection.matrix.base.shape:
                raise AutodiffCompositionError(f"Shape of matrix loaded for '{projection.name}' "
                                               f"({np.array(matrix).shape}) "
                                               f"does not match its shape ({projection.matrix.base.shape})")
            projection.matrix.base = matrix
            projection.parameters.matrix.set(matrix, context=context, override=True)
            projection.parameter_ports['matrix'].parameters.value.set(matrix, context=context, override=True)

    def _get_state_ids(self):
        return super()._get_state_ids() + ["optimizer"]

    def _get_state_struct_type(self, ctx):
        comp_state_type_list = ctx.get_state_struct_type(super())
        pytorch_representation = self._build_pytorch_representation(context=self._context_for_pytorch)
        optimizer_state_type = pytorch_representation._get_compiled_optimizer()._get_optimizer_struct_type(ctx)

        return pnlvm.ir.LiteralStructType((
            *comp_state_type_list,
            optimizer_state_type))

    def _get_state_initializer(self, context):
        comp_states = super()._get_state_initializer(context)
        optimizer_states = tuple()

        return (*comp_states, optimizer_states)

    if torch_available:
        @handle_external_context(fallback_most_recent=True)
        def copy_torch_param_to_projection_matrix(self,
                                                  projection:Union[str, MappingProjection],
                                                  torch_param:Union[torch.nn.Parameter, torch.Tensor, str, int],
                                                  torch_module:torch.nn.Module=None,
                                                  torch_slice:slice=None,
                                                  validate:bool=True,
                                                  context:Optional[Union[Context, str]]=None)->np.ndarray:
            """Assign torch Parameter to `matrix <MappingProjection.matrix>` Parameter of specified `MappingProjection`.
            Return torch_param as the np.ndarray assigned to `matrix <MappingProjection.matrix>` Parameter of
            **projection**.

            Arguments
            ---------

            projection : str or MappingProjection
               specifies `MappingProjection` to which the torch_param is assigned as its `matrix
               <MappingProjection.matrix>` Parameter;  if specified as a str, it must be the name of a
               MappingProjection in the AutodiffComposition.

            torch_param : torch.nn.Parameter, str or int
               specifies torch_param to assign to the `matrix <MappingProjection.matrix>` Parameter of **projection**;
               if it is a torch.nn.Parameter or torch.Tensor, then the **torch_module** argument does not need to be
               specified; if specified as a str or int, it must be the name of a torch Parameter (used to access it in
               the state_dict) or its index (used to access it in the parameterlist) of the **torch_module** argument,
               which must be also specified.

            torch_module : torch.nn.Module : default None
               specifies a torch.nn.Module containing **torch_param** assigned to the`matrix<MappingProjection.matrix>`
               Parameter of **projection**; this does not need to be specified if **torch_param** is a
               torch.nn.Parameter or torch.Tensor, but must be specified if **torch_param** is a str or int.

            torch_slice : slice : default None
               specifies a slice of **torch_param** to assign to the `matrix <MappingProjection.matrix>` Parameter
               of **projection**; if it is not specified, the entire tensor of **torch_param** is used.

              .. warning::
                 **torch_slice** should not be specified if the specification of **torch_param** already takes this
                 into account.

            validate : bool : default True
               specifies whether to validate the **projection** and **torch_param** arguments; setting it to False
               results in more efficient processing if this method is called frequently; however, invalid arguments will
               raise standard Python exceptions rather than more informative AutodiffComposition errors, and unexpected
               results may go unnoticed.

               .. warning::
                  if validate is False, for efficiency: **projection** *must* be a `MappingProjection`, **torch_param**
                  *must* be a torch.Tensor, and both **torch_module** and **torch_slice** are ignored.

            context : Context or None : default most recent Context
               specifies context to use for the value of Projection.matrix;  if it is not provided, then a default
               `Context` is constructed using the `name <Composition.name>` of the AutodiffComposition as the
               `execution_id <Context.execution_id>`, commensurate with the one used bydefault for its `execution
               <AutodiffComposition_Execution>`.
            """
            if validate:
                torch_tensor, projection = self._validate_torch_param_and_projection(torch_param,
                                                                                     torch_module,
                                                                                     torch_slice,
                                                                                     projection)
            else:
                # Assume **torch_param** is passed in as Tensor and **projection** as Projection if validate is False
                torch_tensor = torch_param[torch_slice] if torch_slice else torch_param

            torch_param_as_pnl_matrix = torch_tensor.detach().cpu().numpy().T
            projection.parameters.matrix._set(torch_param_as_pnl_matrix, context)
            projection.parameter_ports['matrix'].parameters.value._set(torch_param_as_pnl_matrix, context)
            return torch_param_as_pnl_matrix

        def copy_projection_matrix_to_torch_param(self,
                                                  projection:Union[str, MappingProjection],
                                                  torch_param:Union[torch.nn.Parameter, torch.Tensor, str, int],
                                                  torch_module:torch.nn.Module=None,
                                                  torch_slice:slice=None,
                                                  validate:bool=True,
                                                  context:Optional[Union[Context, str]]=None)->torch.Tensor:
            """Assign the `matrix <MappingProjection.matrix>` Parameter of a `MappingProjection` to a Pytorch Parameter.

            .. warning:
               If the PyTorch Parameter has requires_grad=True, this will impact its updating in PyTorch.

            Return torch.Tensor assigned to **torch_param**

            Arguments
            ---------

            projection : str or MappingProjection
               specifies `MappingProjection`, the `matrix <MappingProjection.matrix>` of which is assigned torch_param;
               if specified as a str, it must be the name of a MappingProjection in the AutodiffComposition.

            torch_param : torch.nn.Parameter, str or int
               specifies torch Parameter to which the `matrix <MappingProjection.matrix>` of the Projection is assigned;
               if it is a torch.nn.Parameter or torch.Tensor, then the **torch_module** argument does not need to be
               specified; if specified as a str or int, it must be the name of a torch Parameter (used to access it in
               the state_dict) or its index (used to access it in the parameterlist) of the **torch_module** argument,
               which must be also specified.

            torch_module : torch.nn.Module : default None
               specifies a torch.nn.Module containing **torch_param** to which the **projection**'s `matrix
               <MappingProjection.matrix>` Parameter is assigned; this does not need to be specified if **torch_param**
               is a torch.nn.Parameter or torch.Tensor, but must be specified if **torch_param** is a str or int.

            torch_slice : slice : default None
               specifies a slice of **torch_param** to assign to the `matrix <MappingProjection.matrix>` Parameter
               of **projection**; if it is not specified, the entire tensor of **torch_param** is used.

              .. warning::
                 **torch_slice** should not be specified if the specification of **torch_param** already takes this
                 into account.

            validate : bool : default True
               specifies whether to validate the **projection** and **torch_param** arguments; setting it to False
               results in more efficient processing if this method is called frequently; however, invalid arguments
               then raise standard Python exceptions rather than more informative AutodiffComposition errors,
               and unexpected results may go unnoticed.

               .. warning::
                  if validate is False, for efficiency: **projection** *must* be a `MappingProjection`, **torch_param**
                  *must* be a torch.Tensor, and both **torch_module** and **torch_slice** are ignored.

            context : Context or None : default most recent Context
               specifies context to use for the value of Projection.matrix;  if it is not provided, then a default
               `Context` is constructed using the `name <Composition.name>` of the AutodiffComposition as the
               `execution_id <Context.execution_id>`, commensurate with the one used bydefault for its `execution
               <AutodiffComposition_Execution>`.
            """
            if validate:
                torch_tensor, projection = self._validate_torch_param_and_projection(torch_param,
                                                                                     torch_module,
                                                                                     torch_slice,
                                                                                     projection)
            # Assume **torch_param** is passed in as a Tensor and **projection** as a Projection if validate is False
            else:
                torch_tensor = torch_param
            if slice is not None:
                torch_tensor = torch_tensor[torch_slice]
            matrix = projection.parameters.matrix.get(context).T.squeeze()
            matrix_as_tensor = torch.tensor(matrix, dtype=torch_tensor.dtype)
            torch_tensor.data.copy_(matrix_as_tensor)
            return matrix_as_tensor

        def _validate_torch_param_and_projection(self, torch_param, torch_module, torch_slice, projection_spec)->tuple:
            """Validate torch and projection arguments for copying between PyTorch and AutodiffComposition.
            Return tuple of torch.Tensor and MappingProjection.
            """
            method_name = 'copy_torch_param_to_projection_matrix'

            # Torch Parameter specification is a Tensor or a torch.nn.Parameter
            if isinstance(torch_param, torch.Tensor):
                torch_tensor = torch_param

            # Torch Parameter specification is a Tensor or a torch.nn.Parameter
            elif isinstance(torch_param, type(None)):
                if isinstance(torch_module, (torch.nn.Parameter, torch.Tensor)):
                    raise AutodiffCompositionError(f"Specification of 'torch_module' arg in {method_name}() is a "
                                                   f"torch Parameter or Tensor; this should be specified using the "
                                                   f"'torch_para' arg.")
                raise AutodiffCompositionError(f"The 'torch_param' arg in {method_name}() ({torch_param}) must be "
                                               f"specified, using either a torch.nn.Parameter or torch.Tensor, or a "
                                               f"str or int paired with specification of a torch.nn.Module in the "
                                               f"'torch_module' arg.")
            # Torch Parameter specification is a torch.nn.Module
            elif isinstance(torch_param, torch.nn.Module):
                raise AutodiffCompositionError(f"Specification of 'torch_param' arg in {method_name}() ({torch_param}) "
                                               f"is a Module, but must be a torch.nn.Parameter, torch.Tensor, str or "
                                               f"int; if a Module is intended, use the 'torch_module' arg, and specify "
                                               f"the Parameter name or index in the 'torch_param' arg.")

            elif isinstance(torch_param, (str, int)):
                if torch_module is None:
                    raise AutodiffCompositionError(f"Specifying of the 'torch_param' arg in {method_name}() with a "
                                                   f"string or int ({torch_param}) requires the 'torch_module' "
                                                   f"arg to be specified as well.")
                if not isinstance(torch_module, torch.nn.Module):
                    raise AutodiffCompositionError(f"Specification of 'torch_module' arg in {method_name}() "
                                                   f"({torch_module}) must be a torch.nn.Module.")
                if isinstance(torch_param, str):
                    # Name of Parameter was specified, so get it from Module's state_dict,
                    if torch_param not in torch_module.state_dict():
                        raise AutodiffCompositionError(f"'{torch_param}' specified in 'torch_param' arg of "
                                                       f"{method_name}() is not the name of a Parameter in the "
                                                       f"state_dict() for '{torch_module}'.")
                    torch_tensor = torch_module.state_dict()[torch_param]
                else:
                    # Index of Parameter was specified, so get it from Module's parameters() list
                    try:
                        torch_tensor = list(torch_module.parameters())[torch_param]
                    except IndexError:
                        raise AutodiffCompositionError(f"The value ({torch_param}) specified in the 'torch_param' arg "
                                                       f"of {method_name}() is not an index within the range of the "
                                                       f"ParameterList specified for the Module ('{torch_module}').")
            else:
                # Unrecognized specification for torch_param arg.
                raise AutodiffCompositionError(f"Specification of 'torch_param' arg in {method_name}() ({torch_param}) "
                                               f"must be a torch.nn.Parameter, torch.Tensor, str or int.")

            if torch_slice is not None:
                if not isinstance(torch_slice, slice):
                    if isinstance(torch_param, (str, int)):
                        param_ref = f"'{torch_param}'" if isinstance(torch_param, str) else f"{torch_param}"
                        raise AutodiffCompositionError(f"Specification of 'torch_slice' arg in {method_name}() "
                                                       f"('{torch_slice}') for Parameter {param_ref} of {torch_module} "
                                                       f"must be a slice.")
                    else:
                        raise AutodiffCompositionError(f"Specification of 'torch_slice' arg in {method_name}() "
                                                       f"({torch_slice}) must be a slice.")
                torch_tensor = torch_tensor[torch_slice]

            # Parse and validate projection spec
            if projection_spec not in self.projections:
                if isinstance(projection_spec, str):
                    raise AutodiffCompositionError(f"'{projection_spec}' in {method_name}() "
                                                   f"is not the name of a Projection in '{self.name}'.")
                elif isinstance(projection_spec, MappingProjection):
                    raise AutodiffCompositionError(f"'{projection_spec.name}' in {method_name}() "
                                                   f"is not a Projection in '{self.name}'.")
                else:
                    assert False, f"PROGRAM ERROR: Illegal type for 'projection' ({projection_spec}) in {method_name}."
            projection = self.projections[projection_spec]

            torch_param_as_pnl_matrix = torch_tensor.detach().cpu().numpy().T
            bias_note = ""
            if torch_param_as_pnl_matrix.ndim == 1:
                # Note: torch biases are 1d, but PNL requires matrices to be 2d
                torch_param_as_pnl_matrix = np.atleast_2d(torch_param_as_pnl_matrix)
                bias_note = (f" [Note: torch biases, usually 1d, have already been converted to 2d "
                             f"to match PsyNeuLink BIAS Nodes Projections.]")
            if torch_param_as_pnl_matrix.shape != projection.parameters.matrix.get().shape:
                raise AutodiffCompositionError(
                    f"Shape of torch parameter {torch_param_as_pnl_matrix.shape} in {method_name}() does not match "
                    f"shape of matrix for '{projection.name}' {projection.parameters.matrix.get().shape}.{bias_note}")
            return torch_tensor, projection

    def show_graph(self, *args, **kwargs):
        """Override to use PytorchShowGraph if show_pytorch is True"""
        from psyneulink.core.compositions.showgraph import SHOW_LEARNING
        from psyneulink.library.compositions.pytorchshowgraph import SHOW_PYTORCH
        if (SHOW_LEARNING in kwargs and kwargs[SHOW_LEARNING]
                and any(isinstance(node, Composition) for node in self.nodes)
                and (SHOW_PYTORCH not in kwargs or not kwargs[SHOW_PYTORCH])):
            raise AutodiffCompositionError(f"'{self.name}' has a nested Composition, so PyTorch mode must be used "
                                           f"for learning; use 'show_pytorch=True' in the call to show_graph().")
        return self._show_graph.show_graph(*args, **kwargs)

    @property
    def num_learnable_pathways(self):
        """Return number of unique learnable pathways in the AutodiffComposition
        Learnable pathways are ones that end in a non-loss Node and contain at least one learnableMappingProjection;
        Unique learnable pathways are defined as those that have different sets of learnable MappingProjections.
        NOTE: THis method is used to insure that all learnable pathways are assigned a TARGET Node and LossMechanism.
        """
        # Get pathways that end in a non-loss Node and contain at least one learnable MappingProjection
        learning_pathways = [pway for pway in self.pytorch_backprop_pathways
                             if (not isinstance(pway[-1], LossMechanism) and
                                 any(item.learnable for item in pway if isinstance(item, MappingProjection)))]
        # Get learnable Projections in each pway:
        learnable_projs_in_pway = [set(item for item in pway if isinstance(item, MappingProjection) and item.learnable)
                                    for pway in learning_pathways]
        # Reduce to number of unique sets of learnable Projections in learnable pathways
        num_learnable_pathways = len(set(tuple(projs) for projs in learnable_projs_in_pway))

        return num_learnable_pathways

    @property
    def sample_nodes_str(self):
        return [loss_mech.sample for loss_mech in self.loss_mechs_map]

    def target_nodes(self):
        return [loss_mech.target for loss_mech in self.loss_mechs_map]

    @property
    def learning_components(self):
        pytorch_learning_components = (self.get_nodes_by_role(NodeRole.LEARNING_OBJECTIVE)
                                       + self.get_nodes_by_role(NodeRole.TARGET))
        return pytorch_learning_components or super().learning_components

    @property
    def torch_parameters(self):
        """Return Pytorch Parameters for pytorch_representation of AutodiffComposition"""
        try:
            if self.pytorch_representation is None:
                self._build_pytorch_representation()
            return list(self.pytorch_representation.named_parameters())
        except:
            raise AutodiffCompositionError(
                f"PROGRAM ERROR:  problem accessing torch.named_parameters() for '{self.name}'.")

    @property
    def _dummy_projections(self) -> Set['DummyProjection']:
        res = set()
        # NOTE: _dependent_components should possibly be reworked to be
        # a context-dependent method
        for pytorch_repr in self.parameters.pytorch_representation.values.values():
            if pytorch_repr is not None:
                res.update([w.projection for w in pytorch_repr.projection_wrappers])
                try:
                    dummy_proj_pairs = pytorch_repr._projection_wrapper_pairs
                except AttributeError:
                    # currently only GRU wrapper uses them
                    pass
                else:
                    res.update([dummy_proj for dummy_proj, _ in dummy_proj_pairs])
        return res

    @property
    def _dependent_components(self) -> Iterable[Component]:
        res = super()._dependent_components
        res.extend(self._dummy_projections)
        return res

    def _get_default_comp_learning_rate(self):
        self._get_nested_compositions()

    # MODIFIED TEACHER_TARGET NEW:
    @property
    @handle_external_context(fallback_default=True)
    def torch_sample_values(self, context=None):
        if not context.execution_id:
            return None
        pytorch_rep = self.parameters.pytorch_representation._get(context)
        if not pytorch_rep:
            return None
        return [sample.detach().numpy() for samples in pytorch_rep.retained_sample_values for sample in samples]

    @property
    @handle_external_context(fallback_default=True)
    def torch_targets(self, context=None):
        if not context.execution_id:
            return None
        pytorch_rep = self.parameters.pytorch_representation._get(context)
        if not pytorch_rep:
            return None
        return [target.detach().numpy() for targets in pytorch_rep.retained_targets for target in targets]

    @property
    @handle_external_context(fallback_default=True)
    def torch_losses(self, context):
        if not context.execution_id:
            return None
        pytorch_rep = self.parameters.pytorch_representation._get(context)
        if not pytorch_rep:
            return None
        return pytorch_rep.retained_losses
    # MODIFIED TEACHER_TARGET END
