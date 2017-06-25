User's Guide
============


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


The primary constructs in PsyNeuLink are mechanisms, projections, processes, and systems.

`Mechanism`.  A PsyNeuLink mechanism takes an input, transforms it with a function in some way, and makes the
output available to other mechanisms. Its primary purpose is representational transformation (that is, "information
processing"). Mechanisms can take any form of input, use any function, and generate any form of
output.  PsyNeuLink provides Mechanisms with standard functions (e.g., linear and logistic transforms for scalars and
arrays;  integrator functions, including various forms of drift diffusion processes; matrix algebraic functions;
and evaluation/comparison functions). However a mechanism can easily be customized with any function that can be
written in or called from Python.

`Projection` and `Processes <Process>`.  Mechanisms are composed into processes, by linking them into
pathways with projections.  A projection takes the output of its sender mechanism, and transforms it as necessary to
provide it as input to its receiver mechanism.  The primary purpose of a projection is to convey information from one
component to another, converting the information from its sender into a form that is usable by its receiver.
A process is a linear chain of mechanisms connected by projections.
Projections within a pathway can be recurrent, but they cannot branch (that is done with systems, as described below).
Processes can also be configured for learning, in which the projections are trained to produce a specified output for
each of a given set of inputs.

`System`.  A system is composed of a set of processes.  Typically, the processes of a system overlap -- that is,
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

   Two `Processes <Process>` are shown, both belonging to the same `System <System>`.  Each process has a
   series of `ProcessingMechanisms <ProcessingMechanism>` linked by `MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism (see :ref:`figure in System <System_Full_Fig>` for a more
   complete example, that includes components responsible for control and learning).

..
    Every PsyNeuLink object is a subclass of the `Component` subclass.  Every component has a:
        * ``variable``
            the input to its function, used both as a template for the format of the input, as its default value
        * ``function``
            performs the core computation of a component.
        * ``params``
            dictionary of parameters for the function
        * ``value``
            the output of its the function
        * ``name``
            a string that uniquely identifies it


.. _Component_Hierarchy:

Component Hierarchy
-------------------

PsyNeuLink uses the following primary constructs (illustrated in the `figure <System__Figure>` below):

- `System`
    Set of (potentially interacting) processes, that can be managed by a “budget” of control and trained.

    - `Process`
        Takes an input, processes it through an ordered list of mechanisms and projections, and generates an output.

        - `Mechanism`
            Transforms an input representation into an output representation.
            Parameters determine its operation, under the influence of Projections.
            There are two primary types:

            + `ProcessingMechanism`
                Aggregates the inputs it receives from its afferent Projections, transforms them in some way,
                and provides the result as output to its efferent Projections.

            + `AdaptiveMechanism`
                Uses the input it receives from other Mechanisms to modify the parameters of one or more other
                PsyNeuLink components.  There are three primary types:

                + `LearningMechanism`
                    Modifies the matrix of a MappingProjection.

                + `ControlMechanism`
                    Modifies one or more parameters of other Mechanisms.

                + `GatingMechanism`
                    Modifies the value of one or more InputStates and/or OutputStates of other Mechanisms.

        - `Projection`
            Conveys the output of a Mechanism to another Component. There are two primary types, that correspond
            to the two types of Mechanisms:

            + `PathwayProjection`
                The primary type of PathwayProjection is a `MappingProjection`, that provides the output of one
                ProcessingMechanism to the input of another.

            + `ModuatoryProjection`
                Takes the output of an AdaptiveMechanism and uses it to modify the input, output or parameter of
                another Coomponent.  There are three types of ModulatoryProjections, corresponding to the three
                types of AdaptiveMechanisms:

                + `LearningProjection`
                     Takes a LearningSignal from LearningMechanism and uses it to modify the matrix of a
                     MappingProjection.

                + `ControlProjection`
                     Takes a ControlSignal from ControlMechanism and uses it to modify the parameter of a
                     ProcessingMechanism.

                + `GatingProjection`
                     Takes a GatingSignal from a GatingMechanism and uses it to modulate the input or output of a
                     ProcessingMechanism


PsyNeuLink User's Guide
=======================

* :ref:`User_Guide_Organization_and_Principles`
* :ref:`User_Guide_Components`
* :ref:`User_Guide_Compositions`
* :ref:`User_Guide_Processing`
* :ref:`User_Guid_Modulation`


.. _User_Guide_Organization_and_Principles:

Organization and Basic Principles
---------------------------------

This section provides a overview of the basic objects in PsyNeuLink, and how they are used to construct and run models.
There are two primary kinds of objects in PsyNeuLink:  `Components <Component>` and `Compositions <Composition>`.
Components are objects that perform a specific function, and Compositions are objects
that combine Components into a runnable model.

Components
~~~~~~~~~~

    There are two primary types of Components:  `Mechanisms <Mechanism>` and `Projections <Projection>`.
    For those familiar with block modeling systems, Mechanisms are the "blocks" in PsyNeuLink, and Projections are the
    "links".  Mechanisms take inputs, process them in some way, and generate outputs that can be sent to other
    Mechanisms. Projections are the means of sending information from one Mechanism to another.  There are several
    varieties of Mechanisms and Projections, that serve a range of different functions.  At the highest level, they
    are divided into two types:  ones responsible for processing, and ones responsible for modulation.

    **Processing**
    The Components responsible for processing are `ProcessingMechanisms <ProcessingMechanism>` and
    `PathwayProjection <PathwayProjections>`.  They are used to create pathways that transmit and transform
    information, taking the inputs to a model and generating its outputs. The primary types of ProcessingMechanisms
    are `TransferMechanisms <TransferMechanism>` (that perform a "memory-less" transformation of
    their input), IntegratorMechanisms (that maintain a memory of their prior state, and integrate that with new
    inputs), and ObjectiveMechanisms (that evalute and/or compare different sources of input).







, as well as two other fundamental types of Components
(`States <State>` and `Functions <Function>`), that are described in the section below on
`Components <User_Guide_Components>`.  The other primary type of object, `Composition`, has two primary types:
`Processes <Process>` and `Systems <System>` that allow Compositions of different degrees of size and complexity to
be created.  These are described in the section below on `Compositions <User_Guide_Compnents>`.  In each case,
examples are provided that illustrate how these objects are implemented, and that parallel those used in the
interactive `Tutorial <LINK>`.





Organization:
    Two main types:
        Components
            objects that actually do computation;
                two main types, of which there two types of each:
                    mechanisms:
                        processing
                        adaptive
                    projections:
                        pathway
                        modulatory
                two constituent types (i.e. parts of other types)
                    states (mechanisms and projections)
                    functions (all PNL objects)
       Compositions:
          configure components into functioning processes and systems

Basic principles:
   Composition
   Processing
   Modulation
      Learning (most common but most complicated)
      Control
      Gating

 - list item 1
 ..
 - list item 2
 ..

.. _User_Guide_Components:

Components
----------

Other packages that are much better for such applications are:
`Text description <https://URL>`_


.. _User_Guide_Compositions:

Compositions
------------

.. _User_Guide_Processing:

Processing
----------

.. _User_Guide_Modulation:

Modulation
----------


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
