Conventions and Definitions
===========================

* `Conventions`
* `Definitions`
    * `Components <Definitions_Components>`
    * `Compositions <Definitions_Compositions>`
    * `Execution  <Definitions_Execution>`
* `Component_Hierarchy <Definitions_Component_Hierarchy>`


.. _Conventions:

Conventions
-----------

The following conventions are used for the names of PsyNeuLink objects and their documentation:

  + `Component` (class): names use CamelCase (with initial capitalization);
    the initial mention in a section of documentation is formatted as a link (in colored text)
    to the documentation for that Component;
  ..
  + `attribute` or `method` of a Component:  names use lower_case_and_underscore;
    appear in a `small box` in documentation;
  ..
  + **argument** of a method or function:  names use lower_case_and_underscore; formatted in **boldface**.
  ..
  + KEYWORD: uses *UPPER_CASE_AND_UNDERSCORE*;  italicized in documentation.
  ..
  + Example::

          Appears in boxed inset.


.. _Definitions:

Definitions
-----------

The two primary types of objects in PsyNeuLink are Components (basic building blocks)
and Compositions (combinations of Components that implement a model).

.. _Definitions_Components:

`Components <Component>`
~~~~~~~~~~~

Components are objects that perform a specific function. Every Component has a:

* `function <Component.function>` - performs the core computation of the Component;

* `variable <Component.variable>` - the input to the Component's `function <Component.function>`;

* *parameter(s)* - determine how a Component's `function <Component.function>` operates
  (listed in its `Parameters <Component.Parameters>` class);

* `value <Component.value>` - represents the result of the Component's `function <Component.function>`;

* `name <Component.name>` - string label that uniquely identifies the Component.

Two types of Components are the basic building blocks of PsyNeuLink models, Mechanisms and Projections:

* `Mechanisms <Mechanism>` - takes one or more inputs received from its afferent `Projections <Projection>`,
  uses its `function <Mechanism.function>` to combine and/or transform these in some way, and makes the output
  available to other Components.  There are two primary types of Mechanisms in PsyNeuLink:
  ProcessingMechanisms and ModulatoryMechanisms:

  + `ProcessingMechanism`
      Aggregates the inputs it receives from its afferent Projections, transforms them in some way,
      and provides the result as output to its efferent Projections.

  + `ModulatoryMechanism`
      Uses the input it receives from other Mechanisms to modify the parameters of one or more other
      PsyNeuLink Components.  There are two primary types:

      + `ControlMechanism`
          Modifies the input(s), parameter(s), and/or output(s) of other Mechanisms.

      + `LearningMechanism`
          Modifies the matrix of a `MappingProjection`.


* `Projections <Projection>`.
   A Projection takes the output of a Mechanism, and transforms it as necessary to provide it
   as the input to another Component. There are two types of Projections, that correspond to the two types of
   Mechanisms:

   + `PathwayProjection`
       Used in conjunction with ProcessingMechanisms to convey information along processing pathway.
       The primary type of PathwayProjection is a `MappingProjection`, that provides the output of one
       ProcessingMechanism as the input to another.

   + `ModulatoryProjection`
       Used in conjunction with `ModulatoryMechanisms <ModulatoryMechanism>` to regulate the function of other
       Components. Takes the output of a ModulatoryMechanism and uses it to modify the input, output or parameter
       of another Component.  There are two types of ModulatoryProjections, corresponding to the two
       types of ModulatoryMechanisms:

       + `ControlProjection`
            Takes a ControlSignal from a `ControlMechanism` and uses it to modify the input, parameter or output
            of a ProcessingMechanism.

       + `LearningProjection`
            Takes a LearningSignal from a `LearningMechanism` and uses it to modify the matrix of a
            MappingProjection.


* `Ports <Port>`
   A Port is an object that belongs to a Mechanism, and that it is used to represent it input(s), parameter(s)
   of its function, or its output(s).   There are three types of Ports, one for each type of representation,
   each of which can receive and/or send a combination of PathwayProjections and/or ModulatoryProjections
   (see `ModulatorySignal_Anatomy_Figure`):

   + `InputPort`
       Represents a set of inputs to the Mechanism.
       Receives one or more afferent PathwayProjections to a Mechanism, combines them using its
       `function <Port.function>`, and assigns the result (its `value <Port.value>`)as an item of the Mechanism's
       `variable <Mechanism.variable>`.    It can also receive one or more `modulatory projections
       <ModulatoryProjection>` (`ControlProjection` or `GatingProjection`), that modify the parameter(s) of the Port's
       function, and thereby the Port's `value <Port.value>`.

   + `ParameterPort`
       Represents a parameter of the Mechanism's `function <Mechanism.function>`.  Takes the assigned value of the
       parameter as the `variable <Port.variable>` for the Port's `function <Port.function>`, and assigns the result
       as the value of the parameter of the Mechanism's `function <Mechanism.function>` that is used when the Mechanism
       executes.  It can also receive one or more modulatory `ControlProjections <ControlProjection>`,
       that modify the parameter(s) of the Port's function, and thereby the value of the parameter of the Mechanism's
       `function <Mechanism.function>`.

   + `OutputPort`
       Represents an output of the Mechanism.
       Takes an item of the Mechanism's `value <Mechanism.value>` as the `variable <Port.variable>` for the Port's
       `function <Port.function>`, assigns the result as the Port's `value <OutputPort.value>`, and provides that
       to one or more efferent PathwayProjections.  It can also receive one or more `modulatory projections
       <ModulatoryProjection>` (`ControlProjection` or `GatingProjection`), that modify the parameter(s) of the Port's
       function, and thereby the Port's `value <Port.value>`.

* `Functions <Function>` - the most fundamental unit of computation in PsyNeuLink.  Every `Component` has a Function
  object, that wraps an executable function together with a definition of its parameters, and modularizes it so that
  it can be swapped out for another (compatible) one, or replaced with a customized one.  PsyNeuLink provides a
  library of standard Functions (e.g. for linear, non-linear, and matrix transformation; integration, and evaluation and
  comparison), as well as a standard Application Programmers Interface (API) that can be used to "wrap" any function
  that can be written in or called from Python.

.. _Definitions_Compositions:

Compositions
~~~~~~~~~~~~

Compositions are combinations of Components that make up a PsyNeuLink model.  There are two types of Compositions:
Processes and Systems.

`Processes <Process>`.  A Process is the simplest type of Composition: a linear chain of Mechanisms connected by
Projections.  A Process may have recurrent Projections, but it does not have any branches.

`System`.  A system is a collection of Processes that can have any configuration, and is represented by a graph in
which each node is a `Mechanism` and each edge is a `Projection`.  Systems are generally constructed from Processes,
but they can also be constructed directly from Mechanisms and Projections.


.. _Definitions_Compositions__Figure:

**PsyNeuLink Compositions**

.. figure:: _static/System_simple_fig.jpg
   :alt: Overview of major PsyNeuLink Components
   :scale: 50 %

   Two `Processes <Process>` are shown, both belonging to the same `System <System>`.  Each Process has a
   series of `ProcessingMechanisms <ProcessingMechanism>` linked by `MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism (see `figure in System <System_Full_Fig>` for a more
   complete example, that includes Components responsible for learning, control and gating).


.. _Definitions_Execution:

Execution
~~~~~~~~~

PsyNeuLink Mechanisms can be executed on their own.  However, usually, they are executed when a Composition to which
they belong is run.  Compositions are run iteratively in `rounds of execution`, in which each Mechanism in the
composition is given an opportunity to execute.  By default, each Mechanism in a Composition executes exactly once
per round of execution.  However, a `Scheduler` can be used to specify one or more conditions for each Mechanism
that determine whether it runs in a given round of execution.  This can be used to determine when a Mechanism begins
and/or ends executing, how many times it executes or the frequency with which it executes relative to other
Mechanisms, as well as dependencies among Mechanisms (e.g., that one begins only when another has completed).

Since Mechanisms can implement any function, Projections insure that they can "communicate" with
each other seamlessly, and a Scheduler can be used to specify any pattern of execution among Mechanisms in a
Composition, PsyNeuLink can be used to integrate Mechanisms of different types, levels of analysis, and/or time
scales of operation, composing heterogeneous Components into a single integrated system.  This affords modelers the
flexibility to commit each Component of their model to a form of processing and/or level of analysis that is
appropriate for that Component, while providing the opportunity to test and explore how they interact with one
another in a single system.
