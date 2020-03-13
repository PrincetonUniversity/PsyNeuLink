Quick Reference
===============

Sections
--------
    * `Conventions`
    * `Repository_Organization`
    * `PsyNeuLink_Objects`
        * `Quick_Reference_Overview`
        * `Quick_Reference_Components`
        * `Quick_Reference_Compositions`
    * `Quick_Reference_Scheduling`
    * `Quick_Reference_Logging`
    * `Quick_Reference_Graphic_Displays`
    * `Quick_Reference_Preferences`

.. _Conventions:

Conventions
-----------

The following conventions are used for the names of PsyNeuLink objects and their documentation:

  + Class (type of object): names use CamelCase (with initial capitalization); the initial mention in a section of
    documentation is formatted as a link (in colored text) to the documentation for that class;
  ..
  + `attribute` or `method` of a `Component` or `Composition`:  names use lower_case_and_underscore; appear in a
    `small box` in documentation;
  ..
  + **argument** of a method or function:  names use lower_case_and_underscore; formatted in **boldface** in
    documentation.
  ..
  + KEYWORD: uses *UPPER_CASE_AND_UNDERSCORE*;  italicized in documentation.
  ..
  + Examples::

          Appear in boxed insets.

See `Naming` for conventions for default and user-assigned names of instances.

.. _Repository_Organization:

Repository Organization
-----------------------

The PsyNeuLink "repo" is organized into two major sections:

`Core`
~~~~~~

This contains the basic PsyNeuLink objects (described in the next section) that are used to build models and run
simulations, and is divided into three subsections:  `Components <Quick_Reference_Components>` (the basic building
blocks of PsyNeuLink models), `Compositions <Quick_Reference_Compositions>` (objects used to combine Components into
models), and `Scheduling <Quick_Reference_Scheduling>` (used to control execution of the Components within a
Composition).

`Library`
~~~~~~~~~

This contains extensions based on the Core objects (under `Compositions` and `Components`), and
PsyNeuLink implementations of published models (under `Models`).  The Library is meant to be extended, and used both
to compare different models that address similar neural mechanisms and/or psychological functions, and to integrate
these into higher level models of system-level function.

.. _PsyNeuLink_Objects:

PsyNeuLink Objects
------------------

.. _Quick_Reference_Overview:

Overview
~~~~~~~~

The two primary types of objects in PsyNeuLink are `Components <Component>` (basic building blocks) and `Compositions
<Composition>` (combinations of Components that implement a model).  There are four primary types of Components:
`Functions <Function>` are the basic units of computation in PsyNeuLink -- every other type of Component in PsyNeuLink
has at least one Function, and sometimes more.  They "package" an executable method that is assigned to a Component's
`function <Component.function>` attribute, and used to carry out the computation for which the Component is
responsible.
`Mechanisms <Mechanism>` are the basic units of processing in a PsyNeuLink model. They have one or more Functions that
perform their characteristic operations.
`Ports <Port>` represent the input(s) and output(s) of a Mechanism, and the parameters of its Function(s).  Ports
have Functions themselves, that determine the value of the Port, and that can be used to modulate that value for
learning, control and/or gating.
`Projections <Projection>` are used to connect Mechanisms and/or nested Compositions, transmit information between them,
and to modulate the value of their Ports.
Mechanisms and Projections are combined to make up a `Composition`.  A Composition can also contain nested
Compositions, that receive and/or send Projections to Mechanisms and/or other nested Compositions. The outermost
Composition constitutes a model, that can be executed using its `run <Composition.run>` method.

The sections that follow provide a description of the Component types, Composition, and other basic objects in
PsyNeuLink.

.. _Quick_Reference_Components:

`Components <Component>`
~~~~~~~~~~~~~~~~~~~~~~~~

Components are objects that perform a specific function. Every Component has the following core attributes:

* `function <Component.function>` - performs the core computation of the Component (belongs to a PsyNeuLink Function
  assigned to the Component's `function <Component.function>` attribute);

* `variable <Component.variable>` - the input used for the Component's `function <Component.function>`;

* *parameter(s)* - determine how a Component's `function <Component.function>` operates;

* `value <Component.value>` - represents the result of the Component's `function <Component.function>`;

* `name <Component.name>` - string label that uniquely identifies the Component.

The four types of Components in PsyNeuLink -- Functions, Mechanisms, Projections, and Ports -- are described below:

* `Functions <Function>`
   A Function is the most fundamental unit of computation in PsyNeuLink.  Every `Component` has a Function
   object, that wraps a callable object (usually an executable function) together with attributes for its parameters.
   This allows parameters to be maintained from one call of a function to the next, for those parameters to be subject
   to modulation by `ModulatoryProjections <ModulatoryProjection>` (see below), and for Functions to be swapped out
   for one another or replaced with customized ones.  PsyNeuLink provides a library of standard Functions (e.g. for
   linear, non-linear, and matrix transformations, integration, and comparison), as well as a standard Application
   Programmers Interface (API) or creating new Functions that can be used to "wrap" any callable object that can be
   written in or called from Python.

* `Mechanisms <Mechanism>`
     A Mechanism takes one or more inputs received from its afferent `Projections <Projection>`,
     uses its `function <Mechanism_Base.function>` to combine and/or transform these in some way, and makes the output
     available to other Components.  There are two primary types of Mechanisms in PsyNeuLink:
     ProcessingMechanisms and ModulatoryMechanisms:

     - `ProcessingMechanism`
         Aggregates the inputs it receives from its afferent Projections, transforms them in some way,
         and provides the result as output to its efferent Projections.  Subtypes implement various types of
         opearations, such as intergration and comparison.

     - `ModulatoryMechanism`
         Uses the input it receives from other Mechanisms to modify the parameters of one or more other
         PsyNeuLink Components.  There are two primary types:

         + `ControlMechanism`
             Modifies the parameters, inputs and/or outputs of other Mechanisms.  Subtypes are specialized for
             operations such as optimization (e.g., `OptimizationControlMechanism`) or gating (`GatingMechanism`).

         + `LearningMechanism`
             Modifies the matrix of a `MappingProjection`.  Subtypes are specialized for autoassociative (e.g.,
             Hebbian) learning, and various supervised learning algorithms (e.g., reinforcement and backprogation).

* `Projections <Projection>`
   A Projection takes the output of a Mechanism, and transforms it as necessary to provide it
   as the input to another Component. There are two types of Projections, that correspond to the two types of
   Mechanisms:

   + `PathwayProjection`
       Used in conjunction with ProcessingMechanisms to convey information along a processing pathway.
       There is currently one on type of PathwayProjection:

       + `MappingProjection`
         Takes the value of the `OutputPort` of one Mechanism, and converts it as necessary to provide it as
         the variable for the `InputPort` of another Mechanism.

   + `ModulatoryProjection`
       Used in conjunction with ModulatoryMechanisms to regulate the functioning of one or more other Components.
       Takes the output of a `ModulatoryMechanism` and uses it to modify the input, parameters, and/or output of
       another Component.  There are two primary types of ModulatoryProjections, corresponding to the two
       types of ModulatoryMechanisms (see `figure <ModulatorySignal_Anatomy_Figure>`):

       + `ControlProjection`
            Takes a ControlSignal from a `ControlMechanism` and uses it to modify the input, parameter and/or output
            of a ProcessingMechanism.  A `GatingProjection` is a subtype,that is specialized for modulating the input
            or output of a Mechanism.

       + `LearningProjection`
            Takes a LearningSignal from a `LearningMechanism` and uses it to modify the matrix of a
            MappingProjection.

* `Ports <Port>`
   A Port is a Component that belongs to a `Mechanism` or a `Projection`.  For Mechanisms, it is used to represent it
   input(s), the parameter(s) of its function, or its output(s) (see `figure <Mechanism_Figure>`).  For Projections,
   it is used to represent the the parameter(s) of its function.  There are three types of Ports, one for each type of
   information, as described below.  A Port can receive and/or send `PathwayProjections <PathwayProjection>`
   and/or `ModulatoryProjections <ModulatoryProjection>`, depending on its type (see `figure
   <ModulatorySignal_Anatomy_Figure>`).

   + `InputPort`
       Represents a set of inputs to a Mechanism.
       Receives one or more afferent PathwayProjections to a Mechanism, combines them using its `function
       <Port_Base.function>`, and assigns the result (its `value <Port_Base.value>`)as an item of the Mechanism's
       `variable <Mechanism_Base.variable>`.  It can also receive one or more `ControlProjections <ControlProjection>`
       or `GatingProjections <GatingProjection>`, that modify the parameter(s) of the Port's function, and thereby the
       Port's `value <Port_Base.value>`.

   + `ParameterPort`
       Represents a parameter of the `function <Component.function>` of a Mechanism or Projection.  Takes the
       assigned value of the parameter as the `variable <Port_Base.variable>` for the Port's `function <Port_Base
       .function>`, and assigns the result as the value of the parameter used by the Mechanism's `function
       <Mechanism_Base.function>` when the Componet executes.  It can also receive one or more `ControlProjections
       <ControlProjection>` that modify parameter(s) of the Port's `function <Port_Base.function>, and thereby the
       value of the parameters used by the `function <Component.function>` of the Mechanism or Projection.

   + `OutputPort`
       Represents an output of a `Mechanism`.
       Takes an item of the Mechanism's `value <Mechanism_Base.value>` as the `variable <Port_Base.variable>` for the
       Port's `function <Port_Base.function>`, assigns the result as the Port's `value <OutputPort.value>`, and
       provides that to one or more efferent PathwayProjections.  It can also receive one or more `ControlProjections
       <ControlProjection>` or `GatingProjections <GatingProjection>`, that modify parameter(s) of the Port's
       function, and thereby the Port's `value <Port_Base.value>`.


.. _Quick_Reference_Compositions:

`Compositions <Composition>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Composition is made up of one more Mechanisms and/or nested Compositions, connected by Projections.  A Composition
is created by first calling its constructor and then using its `add methods <Composition_Creation>` to add Components.
Every Composition has a `graph <Composition.graph>` attribute, in which each Mechanism or nested Composition is a
node and each `Projection` is a directed edge.  The graph defines the default flow of computation (from each node to
the ones to which it projects)when it is executed using its `run <Composition.run>` method, that may be modified by
`Conditions <Condition>` assigned to the Composition's `scheduler <Composition.scheduler>` (see
`below <Quick_Reference_Scheduling>`.  The graph can be displayed using the Composition's `show_graph
<Composition.show_graph>` method, an example of which is shown in the following figure:

.. _Quick_Reference_Compositions__Figure:

.. figure:: _static/QuickReference_Composition_fig.svg
   :width: 75%

   **Composition.** Example of a PsyNeuLink Composition that contains various types of `Mechanisms <Mechanism>`
   (shown as ovals, which each type shown in parentheses belown the Mechanism's name) and `Projections <Projections>`
   between them (shown as arrows);  see `Basics And Primer <BasicsAndPrimer_Elaborate_Configurations>` for a more
   complete description of the model implemented by this Composition).

.. _Quick_Reference_Scheduling:

`Scheduling <Scheduler>`
------------------------

PsyNeuLink Mechanisms can be executed on their own.  However, usually they are executed as part of a Composition to
which they belong, when that is executed using its `run <Composition.run>` method.  When a Composition is `run
<Composition_Run>`, its Components are executed under the control of its `scheduler <Composition.scheduler>`.  This
is a `Scheduler`, that executes the Composition iteratively in rounds of execution referred to as a `PASS`, in which
each node (Mechanism and/or nested Composition) in the Composition's `graph <Composition.graph>` is given an
opportunity to execute.  By default, each node executes exactly once per `PASS`, in the order determined by the edges
(`Projections <Projection>`) between them.  However, the Scheduler can be used to specify one or more `Conditions
<Condition>` for each node that determine whether it executes in a given `PASS`.  This can be
used to determine when a node begins and/or ends executing, how many times it executes or the frequency with
which it executes relative to other nodes, and any other dependency that can be expressed in terms of the
attributes of other Components in PsyNeuLink. Using a `Scheduler` and a combination of `pre-specified
<Condition_Pre_Specified>` and `custom <Condition_Custom>` Conditions, any pattern of execution can be configured
that is logically possible.

A Composition continues to execute `PASS`\es until its `TRIAL` `termination Condition
<Scheduler_Termination_Conditions>` is met, which constitutes a `TRIAL` of executions.  This is associated with a
single input to the Composition. Multiple `TRIAL` s (corresponding to a sequences of inputs) can be executed using a
Composition's `run <Composition.run>` method.

.. _Quick_Reference_Logging:

Logging
-------

PsyNeuLink supports logging of any attribute of any `Component` or `Composition` under various specified
conditions.  `Logs <Log>` are dictionaries, with an entry for each Component being logged.  The key for each entry is
the name of the Component, and the value is a record of the Component's `value <Component.value>` recorded under the
conditions specified by its `logPref <Component.logPref>` attribute, specified as a `LogLevel`; each record is a
tuple, the first item of which is a time stamp (the `TIME_STEP` of the `RUN`), the second a string indicating the
context in which the value was recorded, and the third the `value <Component.value>` itself.

.. _Quick_Reference_Graphic_Displays:

Graphic Display
---------------

At the moment, PsyNeuLink has limited support for graphic displays:  the graph of a `Composition` can be displayed
using its `show_graph <Composition.show_graph>` method.  This can be used to display just the processing components
(i.e., `ProcessingMechanisms <ProcessingMechanism>` and `MappingProjections <MappingProjection>`), or to include
`learning <LearningMechanism>` and/or `control <ControlMechanism>` components.  A future release will include
a more complete and interacdtive graphical user interface.

.. _Quick_Reference_Preferences:

Preferences
-----------

PsyNeuLink supports a hierarchical system of `Preferences` for all Components and Compositions.  Every object has its
own set of preferences, as does every class of object.  Any preference for an object can be assigned its own value, or
the default value for any of its parent classes for that preference (e.g., an instance of a `DDM` can be assigned
its own preference for reporting, or use the default value for `ProcessingMechanisms <ProcessingMechanism>`,
`Mechanisms <Mechanism>`, or `Components <Component>`.  There are preferences for reporting (i.e., which results of
processing are printed to the console during execution), logging, levels of warnings, and validation (useful for
debugging, but suppressible for efficiency of execution).
