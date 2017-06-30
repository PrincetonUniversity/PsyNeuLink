.. PsyNeuLink documentation master file, created by
   sphinx-quickstart on Wed Oct 19 11:51:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PsyNeuLink Documentation
========================

* `Purpose`
* `Overview`
* `What PsyNeuLink is NOT <What_PsyNeuLink_is_NOT>`
* `Conventions and Definitions <ConventionsAndDefinitions>`
* `Installation`
* `Contributors`
* `Indices_and_Tables`


.. _Purpose:

Purpose
-------

PsyNeuLink is a "block modeling system" for cognitive neuroscience.  Block modeling systems allow components to be
constructed that implement various, possibly disparate functions, and then link them together into a system to
examine how they interact.  In PsyNeuLink, components are used to implement the function of brain subsystems and/or
psychological processes, the interaction of which can then be simulated at the system level.

PsyNeuLink is open source, and meant to be extended. Its goal is to provide an environment for implementing models
of mind/brain function that are modular, customizable, extensible, disseminable, and clearly documented.  It does this
in a manner that:

 - is *computationally general* -- that is, that can implement any desired mechanism or process;
 ..
 - adheres as closely as possible to the insights and design principles that have been learned in computer science
   (e.g., function-based, object-oriented programming);
 ..
 - expresses (as few as possible) *commitments* that reflect general principles of how the mind/brain is organized
   and operates, without committing to any particular detailed model or theory;
 ..
 - expresses these commitments in a form that is powerful, easy to use, and familiar to cognitive neuroscientists;
 ..
 - allows models to be simply and flexibly implemented, using a minimum of coding, and that provides seamless
   integration of, and interaction among disparate components that can vary in their:
     - granularity of representation and function
     - time-scale of operation
 ..
 - provides a standard and accessible environment for model comparison, sharing, and documentation;
 ..
 - has an interface (API) that allows it to be used with other powerful tools for implementing individual components,
   such as:

   * MatLab (general purpose mathematical tools);
   * TensorFlow, Teano (ODE's, deep learning);
   * Emergent (broad class of neurally-plausible connectionist models);
   * ACT-R (symbolic, production system models).

The goal is to encourage users to think about information processing in a "mind/brain-like" way, while imposing as few
constraints as possible on what it is possible to implement or ask the model to do.

.. _What_PsyNeuLink_is_NOT:

What PsyNeuLink is **NOT**
--------------------------

While PsyNeuLink is well suited to the creation of simple to moderately complex models, and to the integration of
disparate existing models into a single, integrated system in which interactions among them can be examined, it is
*not* currently as well suited to other kinds of efforts, such as:

 - extensive model fitting
 - large scale simulations
 - biophysically-realistic models of individual neurons

Other packages that are better suited to such applications are:
`Emergent <https://grey.colorado.edu/emergent/index.php/Main_Page>`_ and
`TensorFlow <https://www.tensorflow.org>`_ (for neural network models);
`HDDM <http://ski.clps.brown.edu/hddm_docs/>`_ (for Drift Diffusion Models);
`ACT-R <http://act-r.psy.cmu.edu>`_ (for production system models);
`Genesis <http://www.genesis-sim.org>`_,
`Neuron <https://www.neuron.yale.edu/neuron/>`_,
and `Nengo <http://www.nengo.ca>`_  (for biophysically-realistic models of neuronal function).
These packages are good for elaborate and detailed models of a particular form.
In contrast, the focus in designing and implementing PsyNeuLink has been to make it as flexible and easy to use as
possible, with the ability to integrate components constructed in other packages (including some of the ones listed
above) into a single functioning system.  These are characteristics that are often (at least in the initial
stages of development) in tension with efficiency (think:  interpreted vs. compiled).  One of the goals for future
development is to make PsyNeuLink more computationally efficient.  At present, however, it is best suited to
developing simpler models, or taking complex or highly detailed models that have been developed --
or subjected to extensive parameter fitting -- in other frameworks, and re-expressing them in a form that is amenable
to integration, documentation, and dissemination.

.. _Overview:

Overview and "Sampler"
----------------------

PsyNeuLink is written in Python, and conforms to the syntax and coding standards for the language.
The secionts below provide some examples of what PsyNeuLink models look like and some of its capabilities.
`Structure_Basic_Constructs` provides an overview of PsyNeuLink Components, Compositions, and their execution.
The `UsersGuide` provides a more thorough description of how PsyNeuLink is organized and its basic principles of
operation.  The `Tutorial` provides an interactive guide to the construction of models using PsyNeuLink.

Basics
~~~~~~

PsyNeuLink models are made of `Components <Component>` and `Compositions <Composition>`:
Components are objects that perform a specific function, and Compositions are used to combine Components into an
executable model.  There are two primary kinds of Components:  `Mechanisms <Mechanism>` and `Projections <Projection>`.
For those familiar with block modeling systems, Mechanisms are the "blocks" in PsyNeuLink, and Projections are the
"links".  Mechanisms take inputs, process them in some way, and generate outputs that can be sent to other
Mechanisms. Projections are used to send information from one Mechanism to another.  A `Composition` uses Projections
to link Mechanisms together into pathways that can execute a process, and processes can be combined to form networks or
circuits that make up a systems-level model.

Mechanisms and Projections fall into two broad categories:  ones that *directly transmit and transform* information,
taking the inputs to a model and generating its outputs;  and ones that *modulate* the transmission and transformation
of information.  PsyNeuLink provides a library of Components of each type.  For example, there is a variety of
ProcessingMechanisms that can be used to transform, integrate, and evaluate information; and there
LearningMechanisms, ControlMechanisms, and GatingMechanism that can be used to modulate those processes.


.. _Index_Simple_Configurations:

Simple Configurations
~~~~~~~~~~~~~~~~~~~~~

Mechanisms can be executed on their own (to gain familiarity with their functions), linked in simple configurations
(for testing isolated interactions), or in Compositions to implement a full model.
Linking Mechanisms for execution can be as simple as placing them in a list -- PsyNeuLink provides the necessary
Projections that connects each to the next one in the list.  For example, the following script uses a simple form of
Composition -- a `Process` -- to create a 3-layered 5-2-5 encoder network, the first layer of which uses a Linear
function (the default for a TransferMechanism), and the other two of which use a LogisticFunction::

    input_layer = TransferMechanism(size=5)
    hidden_layer = TransferMechanism(size=2, function=Logistic)
    output_layer = TransferMechanism(size=5, function=Logistic)
    my_encoder = process(pathway=[input_layer, hidden_layer, output_layer])

Each of the Mechanisms can be executed individually, by simply calling its execute method with an input array::

    output_layer.execute([0, 2.5, 10.9, 2, 7.6])

The full process can be run simply by calling its run method::

    my_encoder.run([0, 2.5, 10.9, 2, 7.6])

The order of that the Mechanisms appear in the list determines the order of their Projections, and PsyNeuLink
picks sensible defaults when necessary Components are not specified.  In the example above, since no projections were
specified, PsyNeuLink automatically created ones that were properly sized to connect each pair of mechanism,
using random initial weights.  However, it is easy to spcify them explicitly, simply by inserting them in between the
 mechanisms in the pathway for the process::

    my_projection_1 = MappingProjection(matrix=(.2 * np.random.rand(2, 5)) + -.1))
    my_encoder = process(pathway=[input_layer, my_projection_1, hidden_layer, output_layer])

The first line above creates a Projection with a 5x2 matrix of random weights constrained to be between -.1 and +.1,
which is then inserted in the pathway between the ``input_layer`` and ``output_layer``.  The matrix itself could also
have been inserted directly, as follows::

    my_encoder = process(pathway=[input_layer, (.2 * np.random.rand(2, 5)) + -.1)), hidden_layer, output_layer])

PsyNeuLink knows to create a MappingProjection using the matrix.  PsyNeuLink is also flexible.  For example,
a recurrent Projection from the ``output_layer`` back to the ``hidden_lalyer`` can be added simply by adding another
entry to the pathway::

    my_encoder = process(pathway=[input_layer, hidden_layer, output_layer, hidden_layer])

This tells PsyNeuLink to create a Projection from the output_layer back to the hidden_layer.  The same could have also
been accomplished by explicilty creating the recurrent connection:

    my_encoder = process(pathway=[input_layer, hidden_layer, output_layer])
    MappingProjection(sender=output_layer,
                      receiver=hidden_layer)

.. _Index_Elaborate_Configurations:

More Elaborate Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuring more complex features is just as simple and flexible.  For example, the feedforward network above can be
trained using backpropagation simply by adding an argument to the constructor for the Process::

    my_encoder = process(pathway=[input_layer, hidden_layer, output_layer], learning=ENABLED)

and then specifying the target for each trial when it is run (here five trials of inputs and targets are specified)::

    my_encoder.run(input=[[0, 0, 0, 0, 0],[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
                   target=[[0, 0, 0, 0, 0],[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

`Backpropation <BackPropagation>` is the default learning method, but PsyNeuLink also currently supports
`Reinforcement Learning <Reinforcement>`, and others are currently being implemented (including Hebbian, Temporal
Differences, and supervised learning for recurrent networks).

PsyNeuLink can also be used to construct models with different kinds of Mechanisms.  For example, the script below
uses a `System` -- a more powerful form of Composition -- to create two feedforward networks that converge on a single
output layer, which combines the inputs and projects to a drift diffusion mechanism (DDM) that decides the response::

    colors_input_layer = TransferMechanism(size=2, function=Logistic, name='COLORS INPUT')
    words_input_layer = TransferMechanism(size=2, function=Logistic, name='WORDS INPUT')
    differencing_weights = np.array([[1], [-1]])
    output_layer = TransferMechanism(size=1, name='OUTPUT')
    decision_mech = DDM(name='DECISION')
    colors_process = process(pathway=[colors_input_layer, differencing_weights, output_layer])
    words_process = process(pathway=[words_input_layer, differencing_weights, output_layer])
    decision_process = process(pathway=[output_layer, decision_mech])
    my_simple_Stroop = system(processes=[colors_process, words_process, decision_process])

As a Composition get more complex, it helps to visualize it.  PsyNeuLink has built-in methods for doing so.
For example, calling ``my_simple_Stroop.show_graph()`` produces the following display:

.. _Index_Simple_Stroop_Example_Figure:

**Composition Graph**

.. figure:: _static/Simple_Stroop_Example_fig.svg

   Graph representation of the Composition in the example above.

As the name of the ``show_graph()`` method suggests, Compositions are represented in PsyNeuLink as graphs, using a
standard dependency dictionary format, so that they can also be submitted to other graph theoretic packages for
display and/or analysis (such as `NetworkX <https://networkx.github.io>`_ and `igraph <http://igraph.org/redirect
.html>`_).

.. _Index_Dynamics_of_Execution:

Dynamics of Execution
~~~~~~~~~~~~~~~~~~~~~

Finally, perhaps the most powerful feature of PsyNeuLink is its ability to simulate models with Components
that execute at arbitrary and disparate "time scales". For example, a Composition can include some Mechanisms
that require fine-grained updates (e.g., Euler integration of a drift diffusion process) with ones that carry out
"single shot" computations (e.g., a single pass through a feedforward neural network). By default, when a Composition
is run, each Component in it is executed at least once.  However, PsyNeuLink has a `Scheduler` that can be used to
design more complex dynamics of execution by assigning one or more `Conditions` to any Mechanism. Conditions can
specify the isolated behavior of a Mechanism (e.g., how many times it should be executed in each
`round of execution <LINK>`), or its behavior relative to that of one or more other Components (e.g., how many times
it should execute or when it should stop executing relative to other Mechanisms).

For example, the following script implements a Composition that integrates a 3-layered feedforward network for
performing a simple stimulus-response mapping task, with a recurrent network that receives input from and feeds back
to the feed-forward network, to provide a simple form of maintained context.  To allow the recurrent layer to settle
following the presentation of each stimulus (which is not required for the feedforward network), the Scheduler can
be used to execute the recurrent layer multiple times but the feedforward network only once in each round execution,
as follows::

    input_layer = TransferMechanism(size = 10)
    hidden_layer = TransferMechanism(size = 100)
    output_layer = TransferMechanism(size = 10)
    recurent_layer = RecurrentTransferMechanism(size = 10)

    feed_forward_network = process(pathway=[input_layer, hidden_layer, output_layer])
    recurrent_network = process(pathway=[hidden_layer, recurrent_layer, hidden_layer])
    full_model = system(processes=[feed_forward_network, recurrent_network])

    my_scheduler = Scheduler(system=full_model)
    my_scheduler.add_condition(my_hidden_layer, Any(EveryNCalls(my_input_layer, 1),
                                                    EveryNCalls(my_recurrent_layer, 10)))
    my_scheduler.add_condition(my_output_layer, EveryNCalls(my_hidden_layer, 2))

The two Conditions added to the controller specify that: 1) ``my_hidden_layer`` should execute whenever either
``input_hidden_layer`` has executed once (to encode the stimulus and make available to the ``recurrent_layer``), and
when the ``recurrent_layer`` has executed 10 times (to allow it to settle on a context representation and
provide that back to the ``hidden_layer``); 2) the ``output_layer`` should execute only after the ``hidden_layer``
has executed twice (to integrate its inputs from both ``input_layer`` and ``recurrent_layer``).

More sophisticated Conditions can also be created.  For example, the ``recurrent_layer`` can be scheduled to
execute until the change in its value falls below a specified threshold as follows::

    minimal_change = lambda mech, thresh : abs(mech.value - mech.previous_value) < thresh))
    my_scheduler.add_condition(my_hidden_layer, Any(EveryNCalls(my_input_layer, 1),
                                                    EveryNCalls(my_recurrent_layer, 1))
    my_scheduler.add_condition(my_recurrent_layer, Any(my_hidden_layer, 1
                                                       Until(minimal_change, my_recurrent_mech, thesh)))

Here, the criterion for stopping execution is defined as a function (``minimal_change``), that is used in an `Until`
Condition.  Any arbitrary Conditions can be created and flexibly combined to construct virtually any schedule of
execution that is logically sensible.


The `User's Guide <UserGuide>` provides a more detailed review of PsyNeuLink's organization and capabilities,
and the `Tutorial` provides an interactive introduction to its use.

.. _Installation:

Installation
------------

Currently, PsyNeuLink is in an alpha state and is not available through pypi/pip.
Instead, you can clone the github repo (https://github.com/PrincetonUniversity/PsyNeuLink).
Clone the master branch.
Download the package with the green "Clone or download" button on the right side of the page and "Download ZIP."

Alternatively, if you are familiar with git, the directory can be cloned as usual through the terminal.
Note: The repo is currently private, so if the link leads to a dead page, reach out to one of the developers to get acccess.

PsyNeuLink is compatible with any version of python 3, but the tutorial (see below) requires a 3.5 installation with
the latest versions of IPython, jupyter, and matplotlib installed.

To install the package, navigate to the cloned directory in a terminal,
switch to your preferred python3 environment, then run the command __"pip install ."__
(make sure to include the period and to use the appropriate pip/pip3 command for python 3.5).
All prerequisite packages will be automatically added to your environment.

Once downloaded, a tutorial can be run using the terminal command ``jupyter notebook`` within the root directory
of the PsyNeuLink package.  Once the jupyter notebook opens, within the list of files click on
"PsyNeuLink Tutorial.ipynb".  This will open the tutorial, that will provide any additional information needed to get
started.

If you have trouble installing the package, or run into other problems, please contact psyneulinkhelp@princeton.edu.


.. _Contributors:

Contributors
------------

* **Allie Burton**, Princeton University
* **Jonathan D. Cohen**, Princeton Neuroscience Institute, Princeton University
* **Peter Johnson**, Princeton Neuroscience Institute, Princeton University
* **Kristen Manning**, Princeton Neuroscience Institute, Princeton University
* **Kevin Mantel**, Princeton Neuroscience Institute, Princeton University
* **Ted Willke**, Intel Labs, Intel Corporation
* **Changyan Wang**, Princeton University
* **Nate Wilson**, Princeton Neuroscience Institute, Princeton University

with substantial and greatly appreciated assistance from:

* **Abhishek Bhattacharjee**, Department of Computer Science, Rutgers University
* **Mihai Capota**, Intel Labs, Intel Corporation
* **Bryn Keller**, Intel Labs, Intel Corporation
* **Garrett McGrath**, Princeton Neuroscience Institute, Princeton University
* **Sebastian Musslick**, Princeton Neuroscience Institute, Princeton University
* **Amitai Shenhav**, Cognitive, Linguistic, & Psychological Sciences, Brown University
* **Michael Shvartsman**, Princeton Neuroscience Institute, Princeton University
* **Ben Singer**, Princeton Neuroscience Institute, Princeton University
* **Jan Vesely**, Department of Computer Science, Rutgers University


..
   .. toctree::
      :maxdepth: 1

      System
      Process

   .. toctree::
      :maxdepth: 3

      Mechanism

   .. toctree::
      :maxdepth: 2

      State
      Projection
      Functions
      Run

.. toctree::
   :hidden:

   ConventionsAndDefinitions
   Component
   Mechanism
   Projection
   State
   Function
   Composition
   System
   Process
   Run
   Scheduler


.. _Indices_and_Tables:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
