.. PsyNeuLink documentation master file, created by
   sphinx-quickstart on Wed Oct 19 11:51:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PsyNeuLink Documentation
========================

* :ref:`Purpose`
* :ref:`Overview`
* :ref:`What PsyNeuLink is NOT`
* :ref:`Component Hierarchy <Component_Hierarchy>`
* :ref:`Installation`
* :ref:`Conventions`
* :ref:`Contributors`
* :ref:`Indices_and_Tables`


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

 - is *computationally general* — that is, that can implement any desired mechanism or process;
 ..
 - adheres as closely as possible to the insights and design principles that have been learned in computer science
   (e.g., function-based, object-oriented programming);
 ..
 - expresses (as few as possible) *commitments* that reflect general principles of how the brain/mind is organized
   and operates, without committing to any particular model or theory;
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
 - interface (API) can easily be constructed to other powerful tools for implementing individual components, such as:

   * MatLab (general purpose mathematical tools);
   * TensorFlow, Teano (ODE's, deep learning);
   * Emergent (broad class of neurally-plausible connectionist models);
   * ACT-R (sybmolic, production system models).

The goal is to encourage users to think about information processing in a "mind/brain-like" way, while imposing as few
constraints as possible on what it is possible to implement or ask the model to do.

.. _What_PsyNeuLink_is_NOT:

What PsyNeuLink is **NOT**
--------------------------

PsyNeuLink is not presently well suited to:

 - extensive model fitting
 - large scale simulations
 - elaborate and detailed models of a particular form
 - biophysically-realistic models of individual neurons

Other packages that are much better for such applications are:
`Emergent <https://grey.colorado.edu/emergent/index.php/Main_Page>`_ and
`TensorFlow <https://www.tensorflow.org>`_ (for neural network models);
`HDDM <http://ski.clps.brown.edu/hddm_docs/>`_ (for Drift Diffusion Models);
`ACT-R <http://act-r.psy.cmu.edu>`_ (for production system models);
`Genesis <http://www.genesis-sim.org>`_,
`Neuron <https://www.neuron.yale.edu/neuron/>`_,
and `Nengo <http://www.nengo.ca>`_  (for biophysically-realistic models of neuronal function).

These packages are all better for elaborate and detailed models of a particular form.
In contrast, the initial efforts in the design and implementation of PsyNeuLink have been put into making it as
flexible and easy to use as possible, including its ability to integrate components constructed in other packages
(includling some of the ones listed above).  These are characteristics that are often (at least in the initial
stages of development) in tension with efficiency (think:  interpreted vs. compiled).  As PsyNeuLink grows and matures,
our expectation is that effort will be put into making it more efficient.


.. _Overview:

Overview
--------

PsyNeuLink is written in Python, and conforms to the syntax and (most of the) coding standards for the language.
The primary constructs in PsyNeuLink are mechanisms, projections, processes, and systems.

:doc:`Mechanism`.  A PsyNeuLink mechanism takes an input, transforms it with a function in some way, and makes the
output available to other mechanisms. Its primary purpose is representational transformation (that is, "information
processing"). Mechanisms can take any form of input, use any function, and generate any form of
output.  PsyNeuLink provides mechanisms with standard functions (e.g., linear and logistic transforms for scalars and
arrays;  integrator functions, including various forms of drift diffusion processes; and matrix algebraic functions).
However a mechanism can easily be customized with any function that can be written in or called from Python.

:doc:`Projection` and :doc:`Processes <Process>`.  Mechanisms are composed into processes, by linking them into
pathways with projections.  A projection takes the output of its sender mechanism, and transforms it as necessary to
provide it as input to its receiver mechanism.  The primary purpose of a projection is to convey information from one
mechanism to another, converting the information from its sender into a form that is usable by its receiver.
A process is a linear chain of mechanisms connected by projections.
Projections within a pathway can be recurrent, but they cannot branch (that is done with systems, as described below).
Processes can also be configured for learning, in which the projections are trained to produce a specified output for
each of a given set of inputs.

:doc:`System`.  A system is composed of a set of processes.  Typically, the processes of a system overlap -- that is,
they share one or more mechanisms.  The network of processes in a system is explicitly represented as a graph, in which
each node is a mechanism, and each edge is a projection.  The system's graph is used to execute it, and can be
exported to display and/or analyze it.  Each system can also have a controller, that can be configured to monitor
the output of any of its mechanisms, evaluate this according to any objective function, and used to control any of the
parameters of the mechanisms' function so as to optimize the objective function.

.. note::
   The functionality described under **Execution** below is still being implemented.  At present, all mechanisms
   operate in :keyword:`trial` mode, and some operate in :keyword:`time_step` mode;  however, the coordination and
   has execution of components with mixed time scales has not yet been fully implemented and debugged.

**Execution**.  Mechanisms in PsyNeuLink can be run at one of two time scales.  In :keyword:`trial` mode, the
transformation computed by a mechanism is "complete" (i.e., the result is stationary.  For example, for an
integrator, the full integral is computed in a single step (analytically where possible).  In :keyword:`time_step`
mode, PsyNeuLink approximates "continuous time" by breaking computations into sequential time steps at a specified
level of precision. For example, for an intergrator, it numerically integrates the path at a specified rate. A system
automatically coordinates the execution of its mechanisms, synchronizing those that are operating on different time
scales (e.g., suspending ones using :keyword:`trial mode` until those using  :keyword:`time_step` mode have met
specified criteria.]

Since mechanisms can implement any function, projections insure that they can "communicate" with
each other seamlessly, and systems coordinate execution over different time scales, PsyNeuLink can be used to
integrate mechanisms of different types, levels of analysis, and/or time scales of operation, composing heterogeneous
elements into a single integrated system.  This affords modelers the flexibility to commit each component of their
model to a form of processing and/or level of analysis that is appropriate for that component, while providing
the opportunity to test and explore how they interact with one another in a single system.


.. _System__Figure:

**Major Components in PsyNeuLink**

.. figure:: _static/System_simple_fig.jpg
   :alt: Overview of major PsyNeuLink components
   :scale: 50 %

   Two :doc:`processes <Process>` are shown, both belonging to the same :doc:`system <System>`.  Each process has a
   series of :doc:`ProcessingMechanisms <ProcessingMechanism>` linked by :doc:`MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism (see :ref:`figure in System <System_Full_Fig>` for a more
   complete example, that includes components responsible for control and learning).

..
    Every PsyNeuLink object is a subclass of the :doc:`Component` subclass.  Every component has a:
        * ``variable``
            the input to its function, used both as a template for the format of the input, as its default value
        * ``function``
            performs the core computation of a component.
        * ``params``
            dictionary of parameters for the function
        * ``value``
            the output of its the function


.. _Component_Hierarchy:

Component Hierarchy
-------------------

PsyNeuLink uses the following primary constructs (illustrated in the :ref:`figure <System__Figure>` below):

- :doc:`System`
    Set of (potentially interacting) processes, that can be managed by a “budget” of control and trained.

    - :doc:`Process`
        Takes an input, processes it through an ordered list of mechanisms and projections, and generates an output.

        - :doc:`Mechanism`
            Transforms an input representation into an output representation.
            Parameters determine its operation, under the influence of projections.
            There are two primary types:

            + :doc:`ProcessingMechanism`
                  Aggregates the inputs it receives from other mechanisms or the input to a process or system,
                  transforms them in some way, and provides the result either as input to other mechanisms and/or
                  to the output of a process or system.

            + :doc:`AdaptiveMechanism`
                  Uses the input it receives from other mechanisms  or the input to a process or system to modify the
                  parameters of one or more other PsyNeuLink components.  There are two primary types:

                  + :doc:`LearningMechanism`
                        Uses an error signal it receives to modify the matrix of a MappingProjection.

                  + :doc:`ControlMechanism`
                        Evaluates the output of one or more other mechanisms, and uses this to modify the
                        parameters of those or other mechanisms in the system to which it belongs.

        - :doc:`Projection`
             Takes the output of a mechanism, possibly transforms it, and uses it to determine the operation of
             another mechanism. There are three primary types:

            + :doc:`MappingProjection`
                Takes the output of a sender mechanism, transform it as necessary to be usable by a receiver mechanism,
                and provides it as input to that receiver mechanism.

            + :doc:`LearningProjection`
                 Takes an error signal (scalar or vector, usually the output of a Monitoring Mechanism)
                 and uses it to modulate the parameter of a projection (usually the matrix of a MappingProjection).

            + :doc:`ControlProjection`
                 Takes an allocation (scalar) (usually the output of a ControlMechanism) and uses it to modulate
                 the parameter(s) of a mechanism.

            [+ GatingSignal — Not yet implemented
                 Takes a gating signal source and uses it to modulate the input or output state of a mechanism.


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

.. _Conventions:

Conventions
-----------

The following conventions are used for the names of PsyNeuLink objects and their documentation:

  + `Component` (class): names use CamelCase (with initial capitalization);
    the initial mention in a section documentation is formatted as a link (in colored text)
    to the documentation for that component.
  ..
  + `attribute` or `method` of a component:  names use lower_case_and_underscore; formatted in a `small box`.
  ..
  + **argument** of a method or function:  names use lower_case_and_underscore; formatted in **boldface**.
  ..
  + KEYWORD: use UPPER_CASE_AND_UNDERSCORE;  formatted as simple text.
  ..
  + Example::

          Appear in boxed insets.


.. _Contributors:

Contributors
------------

* **Jonathan D. Cohen**, Princeton Neuroscience Institute, Princeton University
* **Peter Johnson**, Princeton Neuroscience Institute, Princeton University
* **Kristen Manning**, Princeton Neuroscience Institute, Princeton University
* **Kevin Mantel**, Princeton Neuroscience Institute, Princeton University
* **Ted Willke**, Intel Labs, Intel Corporation
* **Nate Wilson**, Princeton Neuroscience Institute, Princeton University

with substantial assistance from:

* **Mihai Capota**, Intel Labs, Intel Corporation
* **Bryn Keller**, Intel Labs, Intel Corporation
* **Garrett McGrath**, Princeton Neuroscience Institute, Princeton University
* **Sebastian Musslick**, Princeton Neuroscience Institute, Princeton University
* **Amitai Shenhav**, Cognitive, Linguistic, & Psychological Sciences, Brown University
* **Michael Shvartsman**, Princeton Neuroscience Institute, Princeton University
* **Ben Singer**, Princeton Neuroscience Institute, Princeton University

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

   System
   Process
   Mechanism
   State
   Projection
   Run
   Component
   Function


.. _Indices_and_Tables:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`