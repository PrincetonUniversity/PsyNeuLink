Conventions and Structure
=========================

* `Conventions <Structure_Conventions>`
* `Basic_Constructs <Structure_Basic_Constructs>`
    * `Components <Structure_Components>`
    * `Compositions <Structure_Compositions>`
    * `Execution  <Structure_Execution>`
* `Component_Hierarchy < <Structure_Component_Hierarchy>>`


.. _Structure_Conventions:

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


.. _Structure_Basic_Constructs:

Basic Constructs
----------------

The two primary types of objects in PsyNeuLink are `Components <Component>` (basic building blocks)
and `Compositions <Composition>` (combinations of Components that implement a model).

.. _Structure_Components:

Components
~~~~~~~~~~

Components are objects that perform a specific function. Every Component has a:

* `function <Component.function>` - performs the core computation of the Component

* `variable <Component.variabe>` - the input to the Component's `function <Component.function>`

* *parameter(s)* - determine how a Component's `function <Component.function>` operates
  (listed in its `user_params <Component.user_params>` dictionary)

* `value <Component.value>` - represents the result of the Component's `function <Component.function>`

* `name <Component.name>` - string label that uniquely identifies the Component.

Two types of Components are the basic building blocks of PsyNeuLink models, Mechanisms and Projections:

* `Mechanisms <Mechanism>` - takes one or more inputs received from its afferent `Projections <Projection>`,
  uses its `function <Mechanism.function>` to combine and/or transform these in some way, and makes the output
  available to other Components.  There are two primary types of Mechanisms in PsyNeuLink:
  ProcessingMechansms and AdapativeMechanisms:

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


`Projections <Projection>`. A Projection takes the output of a Mechanism, and transforms it as necessary to provide it
 as the input to another Component. There are two types of Projections, that parallel the two types of Mechanisms:
`PathwayProjections <PathwayProjection>`, that are used in conjunction with ProcessingMechanisms to convey information
along processing pathways; and `ModulatoryProjections <ModulatoryProjection>`, that are used in conjunction with
AdaptiveMechanisms to regulate the function of other Components.

* `States`


* `Functions` - the most fundamental unit of computation in PsyNeuLink.  Every `Component` has a Function object, that
  wraps an executable function together with a definition of its parameters, and modularizes it so that it can be
  swapped out for another (compatible) one, or replaced with a customized one.  PsyNeuLink provides a library of
  standard Functions (e.g. for linear, non-linear, and matrix transformation; integration, and evaluation and
  comparison), as well as a standard Application Programmers Interface (API) that can be used to "wrap" any function
  that can be written in or called from Python.

.. _Structure_Compositions:

Compositions
~~~~~~~~~~~~

Compositions are combinations of Components that make up a PsyNeuLink model.  There are two types of Compositions:
Processes and Systems.

`Processes <Process>`.  A Process is the simplest type of Composition: a linear chain of Mechanisms connected by
Projections.  A Process may have recurrent Projections, but it does not have any branches.

`System`.  A system is a collection of Processes that can have any configuration, and is represented by a graph in
which each node is a `Mechanism` and each edge is a `Projection`.  Systems are generally constructed from Processes,
but they can also be constructed directly from Mechanisms and Projections.

.. _Structure_Execution:

Execution
~~~~~~~~~

PsyNeuLink Mechanisms can be executed on their own.  However, usually, they are executed when a Composition to which
they belong is run.  Compositions are run iteratively in `rounds of execution`, in which each Mechanism in the
composition is given an opportunity to execute.  By default, each Mechanism in a Composition executes exactly once
per round of execution.  However, a `Scheduler` can be used to specify one or more conditions for each Mechanism
that determine whether it runs in a given round of execution.  This can be used to determine when a Mechanism begins
and/or ends executing, how many times it executes or the frequency with which it exeuctes relative to other
Mechanisms, as well as dependencies among Mechanisms (e.g., that one begins only when another has completed).

Since Mechanisms can implement any function, Projections insure that they can "communicate" with
each other seamlessly, and a Scheduler can be used to specify any pattern of execution among Mechanisms in a
Composition, PsyNeuLink can be used to integrate Mechanisms of different types, levels of analysis, and/or time
scales of operation, composing heterogeneous Components into a single integrated system.  This affords modelers the
flexibility to commit each Component of their model to a form of processing and/or level of analysis that is
appropriate for that Component, while providing the opportunity to test and explore how they interact with one
another in a single system.

.. _System__Figure:

**Major Components in PsyNeuLink**

.. figure:: _static/System_simple_fig.jpg
   :alt: Overview of major PsyNeuLink components
   :scale: 50 %

   Two `Processes <Process>` are shown, both belonging to the same `System <System>`.  Each process has a
   series of `ProcessingMechanisms <ProcessingMechanism>` linked by `MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism (see :ref:`figure in System <System_Full_Fig>` for a more
   complete example, that includes components responsible for learning, control and gating).


.. _Component_Hierarchy:

Component Hierarchy
-------------------






- `Projection`
   Conveys the output of a Mechanism to another Component. There are two primary types, that correspond
   to the two types of Mechanisms:

   + `PathwayProjection`
       The primary type of PathwayProjection is a `MappingProjection`, that provides the output of one
       ProcessingMechanism as the input to another.

   + `ModuatoryProjection`
       Takes the output of an AdaptiveMechanism and uses it to modify the input, output or parameter of
       another Component.  There are three types of ModulatoryProjections, corresponding to the three
       types of AdaptiveMechanisms:

       + `LearningProjection`
            Takes a LearningSignal from a LearningMechanism and uses it to modify the matrix of a
            MappingProjection.

       + `ControlProjection`
            Takes a ControlSignal from a ControlMechanism and uses it to modify the parameter of a
            ProcessingMechanism.

       + `GatingProjection`
            Takes a GatingSignal from a GatingMechanism and uses it to modulate the input or output of a
            ProcessingMechanism
