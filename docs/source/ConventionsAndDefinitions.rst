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
  (listed in its `user_params <Component.user_params>` dictionary);

* `value <Component.value>` - represents the result of the Component's `function <Component.function>`;

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
      PsyNeuLink Components.  There are three primary types:

      + `LearningMechanism`
          Modifies the matrix of a `MappingProjection`.

      + `ControlMechanism`
          Modifies one or more parameters of other Mechanisms.

      + `GatingMechanism`
          Modifies the value of one or more `InputStates <InputState>` and/or `OutputStates <OutputStates>`
          of other Mechanisms.

* `Projections <Projection>`.
   A Projection takes the output of a Mechanism, and transforms it as necessary to provide it
   as the input to another Component. There are two types of Projections, that correspond to the two types of
   Mechanisms:

   + `PathwayProjection`
       Used in conjunction with ProcessingMechanisms to convey information along processing pathway.
       The primary type of PathwayProjection is a `MappingProjection`, that provides the output of one
       ProcessingMechanism as the input to another.

   + `ModulatoryProjection`
       Used in conjunction with AdaptiveMechanisms to regulate the function of other Components.
       Takes the output of an `AdaptiveMechanism` and uses it to modify the input, output or parameter of
       another Component.  There are three types of ModulatoryProjections, corresponding to the three
       types of AdaptiveMechanisms:

       + `LearningProjection`
            Takes a LearningSignal from a `LearningMechanism` and uses it to modify the matrix of a
            MappingProjection.

       + `ControlProjection`
            Takes a ControlSignal from a `ControlMechanism` and uses it to modify the parameter of a
            ProcessingMechanism.

       + `GatingProjection`
            Takes a GatingSignal from a `GatingMechanism` and uses it to modulate the input or output of a
            ProcessingMechanism

* `States <State>`
   A State is a Component that belongs to a `Mechanism` and is used to represent it input(s), the parameter(s)
   of its function, or its output(s).   There are three types of States, one for each type of representation
   (see `Mechanism_Figure`), each of which can receive and/or send `PathwayProjections <PathwayProjection>` and/or
   `ModulatoryProjections <ModulatoryProjection>` (see `ModulatorySignal_Anatomy_Figure`):

   + `InputState`
       Represents a set of inputs to the Mechanism.
       Receives one or more afferent PathwayProjections to a Mechanism, combines them using its
       `function <State.function>`, and assigns the result (its `value <State.value>`)as an item of the Mechanism's
       `variable <Mechanism.variable>`.  It can also receive one or more `GatingProjections <GatingProjection>`, that
        modify the parameter(s) of the State's function, and thereby the State's `value <State.value>`.

   + `ParameterState`
       Represents a parameter of the Mechanism's `function <Mechanism.function>`.  Takes the assigned value of the
       parameter as the `variable <State.variable>` for the State's `function <State.function>`, and assigns the result
       as the value of the parameter used by the Mechanism's `function <Mechanism.function>` when the Mechanism
       executes.  It can also receive one or more `ControlProjections <ControlProjection>` that modify parameter(s)
       of the State's function, and thereby the value of the parameters used by the Mechanism's
       `function <Mechanism.function>`.

   + `OutputState`
       Represents an output of the Mechanism.
       Takes an item of the Mechanism's `value <Mechanism.value>` as the `variable <State.variable>` for the State's
       `function <State.function>`, assigns the result as the State's `value <OutputState.value>`, and provides that
       to one or more efferent PathwayProjections.  It can also receive one or more
       `GatingProjections <GatingProjection>`, that modify parameter(s) of the State's function, and thereby the
       State's `value <State.value>`.

* `Functions <Function>` - the most fundamental unit of computation in PsyNeuLink.  Every `Component` has a Function
  object, that wraps a callable object (usually an executable function) together with attributes for its parameters.
  This allows parameters to be maintained from one call of a ffunction to the next, for those parameters to be subject
  to modulation by `ControlProjections <ControlProjection>`, and for Functions to be swapped out for one another
  or replaced with customized ones.  PsyNeuLink provides a library of standard Functions (e.g. for linear,
  non-linear, and matrix transformations, integration, and comparison), as well as a standard Application Programmers
  Interface (API) or creating new Functions that can be used to "wrap" any callable object that can be written in or
  called from Python.

.. _Definitions_Compositions:

Compositions
~~~~~~~~~~~~

Compositions are combinations of Components that make up a PsyNeuLink model.  There are two primary types of
Compositions:

   + `Processes <Process>`
       One or more `Mechanisms <Mechanism>` connected in a linear chain by `Projections <Projection>`.  A Process can
       have recurrent Projections, but it cannot have any branches.

   + `System`
       A collection of Processes that can have any configuration, and is represented by a graph in which each node is
        a `Mechanism` and each edge is a `Projection`.  Systems are generally constructed from Processes, but they
        can also be constructed directly from Mechanisms and Projections.


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
they belong is run.  Compositions are run iteratively in rounds of execution referred to as `PASS` \es, in which each
Mechanism in the composition is given an opportunity to execute.  By default, each Mechanism in a Composition
executes exactly once per `PASS`.  However, a `Scheduler` can be used to specify one or more `Conditions <Condition>`
for each Mechanism that determine whether it runs in a given `PASS`.  This can be used to determine when
a Mechanism begins and/or ends executing, how many times it executes or the frequency with which it executes relative
to other Mechanisms, and any other dependency that can be expressed in terms of the attributes of other Components
in PsyNeuLink.  Using a `Scheduler` and a combination of `pre-speciffied <Condition_Pre_Specified>` and
`custom <Condition_Custom>` Conditions, any pattern of execution can be configured that is logically possible.


.. _Definitions_Logging:

Logging
~~~~~~~

PsyNeuLink supports logging of any attribute of any Component or Composition at any `TimeScale` of execution.
Logs are dictionaries, with any entry for each attribute being logged.  The key for each entry is the name of
the attribute, and its value is a record of the attribute's value recorded at the frequency specified by the
`TimeScale` parameter for the entry;  each record is a tuple, the first item of which is a time stamp (the
`TIME_STEP` of the `RUN`), and the second is the value of the attribute at that `TIME_STEP`.

.. _Graphic_Displays:

Graphic Displays
~~~~~~~~~~~~~~~~

At the moment, PsyNeuLink has limited support for graphic displays:  the graph of a `System` can be displayed
using its `show_graph` method.  This can be used to display just the processing components (i.e.,
`ProcessingMechanisms <ProcessingMechanism>` and `MappingProjections <MappingProjection>`, or to include
`learning <LearningMechanism>` and/or control <ControlMechanism>` components.  A future release may include
a more complete graphical user interface.


.. _Definitions_Logging:

Preferences
~~~~~~~~~~~

PsyNeuLink supports a hierarchical system of preferences for all Components and Compositions.