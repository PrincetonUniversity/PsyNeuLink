Basics and Sampler
==================

* `BasicsAndSampler_Basics`
* `BasicsAndSampler_Sampler`
    * `BasicsAndSampler_Simple_Configurations`
    * `BasicsAndSampler_Elaborate_Configurations`
    * `BasicsAndSampler_Dynamics_of_Execution`

.. _BasicsAndSampler_Basics:

Basics
------

**Overview**

PsyNeuLink models are made up of `Components <Component>` and `Compositions <Composition>`:
Components are objects that perform a specific function, and Compositions are used to combine Components into a model.
There are two primary kinds of Components:  `Mechanisms <Mechanism>` and `Projections <Projection>`. For those
familiar with block modeling systems, Mechanisms are the "blocks" in PsyNeuLink, and Projections are the
"links".  Mechanisms take inputs, use a function to process them in some way, and generate outputs that can be sent to
other Mechanisms.  Projections allow the output of one Mechanism to be transmitted to another.  `Compositions` combine
these Components into pathways that constitute a `computational graph <https://en.wikipedia.org/wiki/Graph_
(abstract_data_type)>`_, in which the Mechanisms are nodes and Projections are directed edges. Compositions can also be
nodes, so that one Composition can be nested inside another to create more complex, hierarchical models (e.g., of
circuits or pathways within a larger system-level model). A `Scheduler` coordinates the execution of all of the
Components in a model.  By default, it executes them in the order determined by the Projections among the Mechanisms
and/or nested Compositions.  However, individual Components can be assigned one or more pre-specified or customized
`Conditions <Condition>` to handle more complex structures, such as feedback pathways, and/or the execution
of Components at different time scales (e.g., a recurrent network that needs time to settle before passing
information to a decision layer).

**Mechanisms and Projections**

Mechanisms and Projections fall into two broad categories:  `ProcessingMechanisms <ProcessingMechanism>`
*directly transmit* and possibly *transform* information, and are linked by `PathwayProjections
<PathwayProjection>` that transmit the information between them. *AdaptiveMechanisms <AdpativeMechanism>` *modify*
or *modulate* the transmission and transformation of information, by way of `ModulatoryProjections
<ModulatoryProjecdtion>` to the Components they modulate.  PsyNeuLink provides a library of Components of
each type.  For example, there is a variety of ProcessingMechanisms that can be used to transform, integrate, and
evaluate information in various ways (e.g., to implement layers of a feedforward or recurrent neural network, or a
drift diffusion decision process); and there are `ModulatoryMechanisms <ModulatoryMechanism>` and `LearningMechanisms
<LearningMechanism>` that can be used to modulate the functioning of ProcessingMechanisms and modify Projections,
respectively.  Since Mechanisms can implement any function, Projections ensure that they can "communicate" with
each other seamlessly.

Together, these elements allow PsyNeuLink to implement and integrate processes of different types, levels of analysis,
and/or time scales of operation, composing them into a coherent system.  This affords modelers the flexibility to
commit each part of their model to a form of processing and/or level of analysis that is appropriate for that part,
while providing the opportunity to test and explore how they interact with one another at the level of the entire
system.  The figure below provides an example of the kinds of elements available in PsyNeuLink, and some that are
planned for future inclusion.  The `QuickReference` provides a more detailed overview of PsyNeuLink objects and its
other facilities.  In the sections that follow, the Sampler provides some examples of how these are used to construct
models in PsyNeuLink.

.. _BasicsAndSampler_GrandView_Figure:

.. figure:: _static/BasicsAndSampler_GrandView_fig.svg

    **PsyNeuLink Environment**.  Full-colored items are examples of currently implemented elements; dimmed
    items are examples of elements planned for future implementation.


.. _BasicsAndSampler_Sampler:

Sampler
-------

.. _Simple_Configurations:

Simple Configurations
~~~~~~~~~~~~~~~~~~~~~

Mechanisms can be executed on their own (to gain familiarity with their operation), or linked together and run
in a Composition to implement part of, or an entire model. Linking Mechanisms for execution can be as simple as
creating them and then assiging them to a Composition in a list -- PsyNeuLink provides the necessary Projections that
connects each to the next one in the list, making reasonable assumptions about their connectivity.  For example, the
following example creates a 3-layered 5-2-5 neural network encoder network, the first layer of which takes an an
array of length 5 as its input, and uses a `Linear` function (the default for a `TransferMechanism`), and the other
two of which take 1d arrays of the specified sizes and use a `Logistic` function::

    # Construct the Mechanisms:
    input_layer = ProcessingMechanism(size=5)
    hidden_layer = ProcessingMechanism(size=2, function=Logistic)
    output_layer = ProcessingMechanism(size=5, function=Logistic)

    # Construct the Composition:
    my_encoder = Composition()
    my_encoder.add_linear_processing_pathway([input_layer, hidden_layer, output_layer])

Each of the Mechanisms can be executed individually, by simply calling its `execute <Mechanism_Base.execute>` method
with an appropriately-sized input array, for example::

    output_layer.execute([0, 2.5, 10.9, 2, 7.6])
    >> array([[0.5, 0.92414182, 0.99998154, 0.88079708, 0.9994998 ]])

The Composition connects the Mechanisms into a pathway that form a graph, which can be shown using its `show_graph
<Composition.show_graph>` method:

.. _BasicsAndSampler_Simple_Pathway_Example_Figure:

.. figure:: _static/BasicsAndSampler_SimplePathway_fig.svg
   :width: 30%

   **Composition Graph**  Representation of the graph of the simple Composition in the example above.  Note that the
   Input Mechanism for the Composition is colored green (to designate it is an `INPUT` node), and its output
   Mechanism is colored Red (to designate it at a `OUTPUT` node).

The Composition can be run by calling its `run <Composition.run>` method, with an input array appropriately sized for
the first Mechanism in the pathway (in this case, the input_layer)::

    my_encoder.run([1, 4.7, 3.2, 6, 2])
    [array([0.88079707, 0.88079707, 0.88079707, 0.88079707, 0.88079707])]

The order in which Mechanisms appear in the list of the `add_linear_pathway <Composition.add_linear_pathway>`
method determines their order in the pathway.  More complicated arrangements can be created by adding nodes
individually using a Composition's `add_nodes <Composition.add_nodes>` method, and/or by creating intersecting
pathways, as shown in some of the examples further below.

PsyNeuLink picks sensible defaults when necessary Components are not specified.  In the example above no `Projections
<Projection>` were actually specified, so PsyNeuLink automatically created the appropriate types (in this case,
`MappingProjections<MappingProjection>`), and sized them appropriately to connect each pair of Mechanisms. Each
Projection has a `matrix <Projection.matrix>` parameter that weights the connections between the elements of the output
of its `sender <Projection.sender>` and those of the input to its `receiver <Projection.receiver>`.  Here, the
default is to use a `FULL_CONNECTIVIT_MATRIX`, that connects every element of the sender's array to every element of
the receiver's array weight of 1 (a ). However, it is easy to specify a Projection explicitly, including its matrix,
simply by inserting them in between the Mechanisms in the pathway::

    my_projection = MappingProjection(matrix=(.2 * np.random.rand(2, 5)) - .1))
    my_encoder = Composition()
    my_encoder.add_linear_processing_pathway([input_layer, my_projection, hidden_layer, output_layer])

The first line above creates a Projection with a 2x5 matrix of random weights constrained to be between -.1 and +.1,
which is then inserted in the pathway between the ``input_layer`` and ``hiddeen_layer``.  The matrix itself could also
have been inserted directly, as follows::

    my_encoder.add_linear_processing_pathway([input_layer, (.2 * np.random.rand(2, 5)) - .1)), hidden_layer, output_layer])

PsyNeuLink knows to create a MappingProjection using the matrix.  PsyNeuLink is also flexible.  For example,
a recurrent Projection from the ``output_layer`` back to the ``hidden_layer`` can be added simply by adding another
entry to the pathway::

    my_encoder.add_linear_processing_pathway([input_layer, hidden_layer, output_layer, hidden_layer])

This tells PsyNeuLink to create a Projection from the output_layer back to the hidden_layer.  The same could have also
been accomplished by explicitly creating the recurrent connection::

    my_encoder.add_linear_processing_pathway([input_layer, hidden_layer, output_layer])
    recurent_projection = MappingProjection(sender=output_layer,
                      receiver=hidden_layer)
    my_encoder.add_projection(recurent_projection)


.. _BasicsAndSampler_Elaborate_Configurations:

More Elaborate Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuring more complex models is also relatively simple.  For example, the script below implements a model of the
`Stroop task <https://en.wikipedia.org/wiki/Stroop_effect>`_ by creating two feedforward pathways that converge on a
single output layer, which combines the inputs and projects to a drift diffusion mechanism (DDM) that decides the
response::

    # Construct the Mechanisms:
    colors_input_layer = TransferMechanism(size=2, function=Logistic, name='COLORS INPUT')
    words_input_layer = TransferMechanism(size=2, function=Logistic, name='WORDS INPUT')
    output_layer = TransferMechanism(size=1, name='OUTPUT')
    decision_mech = DDM(name='DECISION')

    # Define a weight matrix used to specify the MappingProjection
    # from each of the input layers to the output_layer
    differencing_weights = np.array([[1], [-1]])

    # Construct the model:
    Stroop_model = Composition()
    Stroop_model.add_linear_processing_pathway([colors_input_layer, differencing_weights, output_layer])
    Stroop_model.add_linear_processing_pathway([words_input_layer, differencing_weights, output_layer])
    Stroop_model.add_linear_processing_pathway([output_layer, decision_mech])


In this example, ``differencing_weights`` is used to specify a `MappingProjection` between the input layer of each
pathway and the Mechanism (``output_layer``) on which they converge.

As a Composition gets more complex, it helps to visualize it.  PsyNeuLink has built-in methods for doing so.
For example, calling ``Stroop_model.show_graph()`` produces the following display:

.. _BasicsAndSampler_Simple_Stroop_Example_Figure:

**Composition Graph**

.. figure:: _static/Simple_Stroop_Example_fig.svg

   Representation of the Composition in the example above.

As the name of the ``show_graph()`` method suggests, Compositions are represented in PsyNeuLink as graphs, using a
standard dependency dictionary format, so that they can also be submitted to other graph theoretic packages for
display and/or analysis (such as `NetworkX <https://networkx.github.io>`_ and `igraph <http://igraph.org/redirect
.html>`_).


.. _BasicsAndSampler_Dynamics_of_Execution:

Dynamics of Execution
~~~~~~~~~~~~~~~~~~~~~

Finally, perhaps the most powerful feature of PsyNeuLink is its ability to simulate models with Components
that execute at arbitrary and disparate "time scales". For example, a Composition can include some Mechanisms
that require fine-grained updates (e.g., Euler integration of a drift diffusion process) with ones that carry out
"single shot" computations (e.g., a single pass through a feedforward neural network). By default, when a Composition
is run, each Component in it is executed at least once.  However, PsyNeuLink has a `Scheduler` that can be used to
design more complex dynamics of execution by assigning one or more `Conditions <Condition>` to any Mechanism. Conditions
can specify the isolated behavior of a Mechanism (e.g., how many times it should be executed in each `TRIAL`), or its
behavior relative to that of one or more other Components (e.g., how many times it should execute or when it should
stop executing relative to other Mechanisms).

For example, the following script implements a Composition that integrates a 3-layered feedforward network for
performing a simple stimulus-response mapping task, with a recurrent network that receives input from and feeds back
to the feed-forward network, to provide a simple form of maintained context.  To allow the recurrent layer to settle
following the presentation of each stimulus (which is not required for the feedforward network), the Scheduler can
be used to execute the recurrent layer multiple times but the feedforward network only once in each `TRIAL`, as
follows::

    # Construct the Mechanisms:
    input_layer = TransferMechanism(size = 10)
    hidden_layer = TransferMechanism(size = 100)
    output_layer = TransferMechanism(size = 10)
    recurrent_layer = RecurrentTransferMechanism(size = 10)

    # Construct the Processes:
    feed_forward_network = Process(pathway=[input_layer, hidden_layer, output_layer])
    recurrent_network = Process(pathway=[hidden_layer, recurrent_layer, hidden_layer])

    # Construct the System:
    full_model = System(processes=[feed_forward_network, recurrent_network])

    # Construct the Scheduler:
    my_scheduler = Scheduler(system=full_model)

    # Add Conditions to the Scheduler:
    my_scheduler.add_condition(my_hidden_layer,
                               Any(EveryNCalls(my_input_layer, 1),
                               EveryNCalls(my_recurrent_layer, 10)))
    my_scheduler.add_condition(my_output_layer,
                               EveryNCalls(my_hidden_layer, 2))

The two Conditions added to the Scheduler specify that:

   1. ``my_hidden_layer`` should execute whenever either ``input_hidden_layer`` has executed once (to encode the stimulus and make available to the ``recurrent_layer``), or when the ``recurrent_layer`` has executed 10 times (to allow it to settle on a context representation and provide that back to the ``hidden_layer``)

   2. the ``output_layer`` should execute only after the ``hidden_layer`` has executed twice (to integrate its inputs from both ``input_layer`` and ``recurrent_layer``).

More sophisticated Conditions can also be created.  For example, the ``recurrent_layer`` can be scheduled to
execute until the change in its value falls below a specified threshold as follows::

    # Define a function ``converge`` that detects when a Mechanism has converged such that
    # none of elements has changed more than ``epsilon`` since the last execution
    def converge(mech, thresh):
        for val in mech.delta:
            if abs(val) >= thresh:
                return False
        return True
    epsilon = 0.01

    # Add a Condition to the Scheduler that uses the ``converge`` function to continue
    # executing the ``recurrent_layer`` while it has not (i.e., until it has) converged
    my_scheduler.add_condition(my_hidden_layer,
                               Any(EveryNCalls(my_input_layer, 1),
                               EveryNCalls(my_recurrent_layer, 1)))
    my_scheduler.add_condition(my_recurrent_layer,
                               All(EveryNCalls(my_hidden_layer, 1),
                                   WhileNot(converge, my_recurrent_mech, epsilon)))

Here, the criterion for stopping execution is defined as a function (``converge``), that is used in a `WhileNot`
Condition.  Any arbitrary Conditions can be created and flexibly combined to construct virtually any schedule of
execution that is logically sensible.

.. _BasicsAndSampler_Control:

Control
~~~~~~~

One of the distinctive features of PsyNeuLink is the ability to easily create models that include control;  that is,
Mechanism that can evaluate the output of other Mechanisms (or nested Compositions), and use this to regulate
the processing of those Mechanisms.  For example, the following extension of the Stroop model monitors conflict in
the output_layer of the Stroop model above on each trial, and uses that to determine how much to control to allocate
to the ColorNaming vs. WordReading pathways.

<CONFLICT MONITORING EXAMPLE HERE>

A more elaborate example of this model can be found at `BotvinickConflictMonitoringModel`. More complicated forms of
control are also possible, for example, ones run internal simulations to determine the amount of control to optimize
some criterion

<EVC EXAMPLE HERE>


.. _BasicsAndSampler_Control:

Learning
~~~~~~~~

For example, the feedforward network above can be
trained using backpropagation simply by adding the **learning** argument to the constructor for the Process::

    my_encoder = Process(pathway=[input_layer, hidden_layer, output_layer], learning=ENABLED)

and then specifying the target for each trial when it is executed (here, the Process' `run <Process.run>` command
is used to execute a series of five training trials, one that trains it on each element of the input)::

    my_encoder.run(input=[[0,0,0,0,0], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]],
                   target=[[0,0,0,0,0], [1,0,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]])

`Backpropagation <BackPropagation>` is the default learning method, but PsyNeuLink also currently supports
`Reinforcement Learning <Reinforcement>`, and others are currently being implemented (including Hebbian, Temporal
Differences, and supervised learning for recurrent networks).


-----------------

STUFF TO ADD:

One of the most useful applications for PsyNeuLink is the design of models that include control processes.
XXX USER DEFINED FUNCTIONS
XXX CONTROL (STROOP)
XXX HETEROGENOUS TYPES: ADD DECISION MAKING USING DDM
XXX LEARNING:  USING RL AND BP
XXX NESTED COMPOSITIONS: AUTODIFF
XXX COMPILATION

The `User's Guide <UserGuide>` provides a more detailed review of PsyNeuLink's organization and capabilities,
and the `Tutorial` provides an interactive introduction to its use.
