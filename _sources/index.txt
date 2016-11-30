.. PsyNeuLink documentation master file, created by
   sphinx-quickstart on Wed Oct 19 11:51:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PsyNeuLink Documentation
========================


Overview
--------

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

Every PsyNeuLink object is a subclass of the :doc:`Component` subclass.  Every component has a:
    * ``variable``
        the input to its function, used both as a template for the format of the input, as its default value
    * ``function``
        performs the core comoputation of a component.
    * ``params``
        dictionary of parameters for the function


PsyNeuLink uses the following primary constructs:

- :doc:`System`
    Set of (potentially interacting) processes, possibly managed by a ControlMechanism.

    - :doc:`Process`
        Component that takes an input, processes it through an ordered list of mechanisms (and projections)
        and generates an output.

        - :doc:`Mechanism`
            Component that takes an input, transforms it in some way, and provides
            it as an output that can be used for some purpose.
            ..

            + :doc:`ProcessingMechanism`
                  Component aggregrate the input they receive from other mechanisms in a process or system,
                  and/or the input to a process or system, transform it in some way, and provide the result either as
                  input for other mechanisms and/or the output of a process or system.
            ..
            + :doc:`ControlMechanism`
                  Component that evaluates the output of one or more other mechanisms, and uses this to modify the
                  function parameters of those or other mechanisms.
            ..
            + :doc:`MonitoringMechanism`
                  Component that monitors the output of one or more other mechanisms, receive training (target) values,
                  and compare these to generate error signals used for learning

        - :doc:`Projection`
            Component that takes a source of input from one mechanism (its sender) and conveys it to another (its
            receiver), transforming it if necessary so that it is usable by its receiver.


            + :doc:`MappingProjection`
                Takes the output of a sender mechanism, transform it as necessary to be usable by a receiver mechanism,
                and then provide that as input to the receiver mechanism.

            + :doc:`ControlProjection`
                Takes an allocation (scalar), possibly transforms it,
                and uses it to modulate the parameter(s) of another mechanism's function.

            + :doc:`LearningProjection`
                Takes an error signal (scalar or vector), possibly transforms it,
                and uses it to modulate the parameter(s) of a projection function (e.g., the weights of a matrix).

            [+ GatingSignal — Not yet implemented
                Takes a source, possibly transforms it, and uses it to
                modulate the input or output state of a mechanism.


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