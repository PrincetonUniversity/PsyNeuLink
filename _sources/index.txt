.. PsyNeuLink documentation master file, created by
   sphinx-quickstart on Wed Oct 19 11:51:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PsyNeuLink Documentation
========================

Purpose
-------

PsyNeuLink is a "block modeling system" for cognitive neuroscience.  Block modeling systems allow components to be
constructed that implement various, possibly disparate functions, and then link them together into a system to
examine how they interact.  In PsyNeuLink, components are used to implement the function of brain subsystems and/or
psychological processes, the interaction of which can then be simulated.

PsyNeuLink is open source, and meant to be extended. Its goal is to provide an environment for implementing models
of mind/brain function that are modular, customizable, extensible and disseminable.  It does this in a manner that:

 - is *computationally general* — that is, that can implement any desired mechanism or process;
 ..
 - adheres as closely as possible to the insights and design principles that have been learned in computer science
   (e.g., function-based, object-oriented programming);
 ..
 - expresses (as few as possible) *commitments* that reflect general principles of how the brain/mind is organized
   and functions, without committing to any particular model or theory;
 ..
 - expresses these commitments in a form that is powerful, easy to use, and familiar to cognitive neuroscientists;
 ..
 - allows models to be simply and flexibly implemented, using a minimum of coding, and that provides seamless
   integration of, and interaction among disparate components that can vary in their:
     - time-scale of operation
     - granularity of representation and function
 ..
 - provides a standard environment for model comparison, sharing, and documentation;
 ..
 - provides an interface (API) to other powerful tools for implementing individual components, such as:
   - MatLab (general purpose math tools0;
   - TensorFlow, Teano (deep learning; ODEs);
   - Emergent (broad class of neurally-plausible connectionist models);
   - ACT-R (sybmolic, production system models).

The goal is to encourage users to think about information processing in a "mind/brain-like" way, while imposing as few
constraints as possible on what it is possible to implement or ask a model to do.


Functional Architecture
-----------------------

PsyNeuLink uses the following primary constructs (illustrated in the :ref:`figure <System__Figure>` below):

- :doc:`System`
    Set of (potentially interacting) processes, that can be managed by a “budget” of control and trained.

    - :doc:`Process`
        Takes an input, processes it through an ordered list of mechanisms and projections, and generates an output.

        - :doc:`Mechanism`
            Transforms an input representation into an output representation.
            Parameters determine its operation, under the influence of projections.
            There are three primary types:
            ..

            + :doc:`ProcessingMechanism`
                  Aggregates the inputs it receives from other mechanisms or the input to a process or system,
                  transforms them in some way, and provides the result either as input to other mechanisms and/or
                  to the output of a process or system.
            ..
            + :doc:`ControlMechanism`
                  Evaluates the output of one or more other mechanisms, and uses this to modify the parameters of those
                  or other mechanisms.
            ..
            + :doc:`MonitoringMechanism`
                   Monitors the output of one or more other mechanisms, compares these to a target value,
                   and generates an error signal used for learning.

        - :doc:`Projection`
             Takes the output of a mechanism, possibly transforms it, and uses it to determine the operation of
             another mechanism. There are three primary types:


            + :doc:`MappingProjection`
                Takes the output of a mechanism, transform it as necessary to be usable by a receiver mechanism,
                and provides it as input to that receiver mechanism.

            + :doc:`ControlProjection`
                 Takes an allocation (scalar) (usually the output of a ControlMechanism) and uses it to modulate
                 the parameter(s) of a mechanism.

            + :doc:`LearningProjection`
                 Takes an error signal (scalar or vector, usually the output of a Monitoring Mechanism)
                 and uses it to modulate the parameter of a projection (usually the matrix of a MappingProjection).

            [+ GatingSignal — Not yet implemented
                 Takes a gating signal source and uses it to modulate the input or output state of a mechanism.

.. _System__Figure:

**Major Components in PsyNeuLink**

.. figure:: _static/System_full_fig.png
   :alt: Overview of major PsyNeuLink components
   :scale: 75 %

   Two :doc:`processes <Process>` are shown, both belonging to the same :doc:`system <System>`.  Each process has a
   series of :doc:`ProcessingMechanisms <ProcessingMechanism>` linked by :doc:`MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism.  The latter projects to a
   :doc:`MonitoringMechanism <MonitoringMechanism>` that is used to drive learning in Process B.  It also projects to
   a :doc:`ControlMechanism <ControlMechanism>` that regulates ProcessingMechanisms in both Processes A and B.  Note
   that while the mechanisms and projections responsible for learning belong to, and are restricted to a single
   process, those responsible for control belong to the system and can monitor and/or control mechanisms belonging to
   more than one process.

..
    Every PsyNeuLink object is a subclass of the :doc:`Component` subclass.  Every component has a:
        * ``variable``
            the input to its function, used both as a template for the format of the input, as its default value
        * ``function``
            performs the core computation of a component.
        * ``params``
            dictionary of parameters for the function


Installation
------------

The tutorial is available at/by [???]

Currently, PsyNeuLink is in an alpha state and is not available through pypi/pip.
Instead, you can clone the github repo (https://github.com/PrincetonUniversity/PsyNeuLink).
Clone the master branch.
Download the package with the green "Clone or download" button on the right side of the page and "Download ZIP."

Alternatively, if you are familiar with git, the directory can be cloned as usual through the terminal.
Note: The repo is currently private, so if the link leads to a dead page, reach out to one of the developers to get acccess.

PsyNeuLink is compatible with any version of python 3,
but this tutorial requires a 3.5 installation with the latest versions of IPython, jupyter, and matplotlib installed.

To install the package, navigate to the cloned directory in a terminal,
switch to your preferred python3 environment, then run the command __"pip install ."__
(make sure to include the period and to use the appropriate pip/pip3 command for python 3.5).
All prerequisite packages will be automatically added to your environment.


Contributors
------------

* **Jonathan D. Cohen**, Princeton Neuroscience Institute, Princeton University
* **Peter Johnson**, Princeton Neuroscience Institute, Princeton University
* **Bryn Keller**, Intel Labs, Intel Corporation
* **Sebastian Musslick**, Princeton Neuroscience Institute, Princeton University
* **Amitai Shenhav**, Cognitive, Linguistic, & Psychological Sciences, Brown University
* **Michael Shvartsman**, Princeton Neuroscience Institute, Princeton University
* **Ted Willke**, Intel Labs, Intel Corporation
* **Nate Wilson**, Princeton Neuroscience Institute, Princeton University


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
   Function
   Run


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`