# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  Projection ***********************************************************

"""

Contents
--------
  * `Projection_Overview`
  * `Projection_Creation`
  * `Projection_Structure`
      - `Projection_Sender`
      - `Projection_Receiver`
  * `Projection_Execution`
  * `Projection_Class_Reference`

.. _Projection_Overview:

Overview
--------

Projections allow information to be passed between `Mechanisms <Mechanism>`.  A Projection takes its input from
its `sender <Projection_Base.sender>` and transmits that information to its `receiver <Projection_Base.receiver>`.  The
`sender <Projection_Base.sender>` and `receiver <Projection_Base.receiver>` of a Projection are always `Ports <Port>`:
the `sender <Projection_Base.sender>` is always the `OutputPort` of a `Mechanism <Mechanism>`; the `receiver
<Projection_Base.receiver>` depends upon the type of Projection.  There are two broad categories of Projections,
each of which has subtypes that differ in the type of information they transmit, how they do this, and the type of
`Port <Port>` to which they project (i.e., of their `receiver <Projection_Base.receiver>`):

* `PathwayProjection <PathwayProjection>`
    Used in conjunction with `ProcessingMechanisms <ProcessingMechanism>` to convey information along a processing
    `pathway <Process.pathway>`.  There is currently one on type of PathwayProjection:

  * `MappingProjection`
      takes the `value <OutputPort.value>` of an `OutputPort` of a `ProcessingMechanism <ProcessingMechanism>`
      converts it by convolving it with the MappingProjection's `matrix <MappingProjection.matrix>`
      parameter, and transmits the result to the `InputPort` of another ProcessingMechanism.  Typically,
      MappingProjections are used to connect Mechanisms in the `pathway` of a `Process`, though they can be use for
      other purposes as well (for example, to convey the output of an `ObjectiveMechanism` to a `ModulatoryMechanism
      <ModulatoryMechanism>`).

* `ModulatoryProjection <ModulatoryProjection>`
    takes the `value <OutputPort.value>` of a `ModulatorySignal <ModulatorySignal>` of a `ModulatoryMechanism
    <ProcessingMechanism>`, uses it to regulate modify the `value <Port_Base.value>` of an `InputPort`,
    `ParameterPort` or `OutputPort` of another Component.  ModulatorySignals are specialized types of `OutputPort`,
    that are used to specify how to modify the `value <Port_Base.value>` of the `Port <Port>` to which a
    ModulatoryProjection projects. There are three types of ModulatoryProjections, corresponding to the three types
    of ModulatoryMechanisms (and corresponding ModulatorySignals; see `figure <ModulatorySignal_Anatomy_Figure>`),
    that project to different types of `Ports <Port>`:

  * `LearningProjection`
      takes the `value <LearningSignal.value>` of a `LearningSignal` of a `LearningMechanism`, and transmits this
      to the `ParameterPort` of a `MappingProjection` that uses it to modify its `matrix <MappingProjection.matrix>`
      parameter. LearningProjections are used in the `learning Pathway(s) <Composition_Learning_Pathway>` of a
      `Composition`.
  ..
  * `ControlProjection`
      takes the `value <ControlSignal.value>` of a `ControlSignal` of a `ControlMechanism <ControlMechanism>`, and
      transmit this to the `ParameterPort of a `ProcessingMechanism <ProcessingMechanism>` that uses it to modify
      the parameter of the `Mechanism <Mechanism>` (or its `function <Mechanism_Base.function>`) for which it is
      responsible.
      COMMENT:
      ControlProjections are used in the `control Pathway(s) <Composition_Control_Pathway>` of a `Composition`.
      COMMENT
  ..
  * `GatingProjection`
      takes the `value <GatingSignal.value>` of a `GatingSignal` of a `GatingMechanism`, and transmits this to
      the `InputPort` or `OutputPort` of a `ProcessingMechanism <ProcessingMechanism>` that uses this to modify the
      Port's `value <Port_Base.value>`.  GatingProjections are a special subclass of ControlProjections.

.. _Projection_Creation:

Creating a Projection
---------------------

A Projection can be created on its own, by calling the constructor for the desired type of Projection.  More
commonly, however, Projections are either specified `in context <Projection_Specification>`, or are `created
automatically <Projection_Automatic_Creation>`, as described below.


.. _Projection_Specification:

*Specifying a Projection*
~~~~~~~~~~~~~~~~~~~~~~~~~

Projections can be specified in a number of places where they are required or permitted, for example in the
specification of a `pathway <Process.pathway>` for a `Process`, where the value of a parameter is specified
(e.g., to assign a `ControlProjection`) or where a `MappingProjection` is specified  (to assign it a
`LearningProjection <MappingProjection_Tuple_Specification>`).  Any of the following can be used to specify a
Projection in context:

  * **Constructor** -- used the same way in context as it is ordinarily.
  ..
  * **Projection object** -- must be a reference to a Projection that has already been created.
  ..
  * **Projection subclass** -- creates a default instance of the specified Projection type.  The assignment or creation
    of the Projection's `sender <Projection_Base.sender>` is handled in the same manner as described below for keyword
    specifications.
  ..
  * **Keyword** -- creates a default instance of the specified type, which can be any of the following:

      * *MAPPING_PROJECTION* -- if the `sender <MappingProjection.sender>` and/or its `receiver
        <MappingProjection.receiver>` cannot be inferred from the context in which this specification occurs, then its
        `initialization is deferred <MappingProjection_Deferred_Initialization>` until both of those have been
        determined (e.g., it is used in the specification of a `pathway <Process.pathway>` for a `Process`). For
        MappingProjections, a `matrix specification <MappingProjection_Matrix_Specification>` can also be used to
        specify the projection (see **value** below).
      COMMENT:

      * *LEARNING_PROJECTION*  (or *LEARNING*) -- this can only be used in the specification of a `MappingProjection`
        (see `tuple <MappingProjection_Matrix_Specification>` format).  If the `receiver <MappingProjection.receiver>`
        of the MappingProjection projects to a `LearningMechanism` or a `ComparatorMechanism` that projects to one,
        then a `LearningSignal` is added to that LearningMechanism and assigned as the LearningProjection's `sender
        <LearningProjection.sender>`;  otherwise, a LearningMechanism is `automatically created
        <LearningMechanism_Creation>`, along with a LearningSignal that is assigned as the LearningProjection's `sender
        <LearningProjection.sender>`. See `LearningMechanism_Learning_Configurations` for additional details.
      COMMENT

      # FIX 5/8/20 [JDC] ELIMINATE SYSTEM:  IS IT TRUE THAT CONTROL SIGNALS ARE AUTOMATICALLY CREATED BY COMPOSITIONS?
      * *CONTROL_PROJECTION* (or *CONTROL*) -- this can be used when specifying a parameter using the `tuple format
        <ParameterPort_Tuple_Specification>`, to create a default `ControlProjection` to the `ParameterPort` for that
        parameter.  If the `Component <Component>` to which the parameter belongs is part of a `Composition`, then a
        `ControlSignal` is added to the Composition's `controller <Composition.controller>` and assigned as the
        ControlProjection's `sender <ControlProjection.sender>`;  otherwise, the ControlProjection's `initialization
        is deferred <ControlProjection_Deferred_Initialization>` until the Mechanism is assigned to a Composition, at
        which time the ControlSignal is added to the Composition's `controller <Composition.controller>` and assigned
        as its the ControlProjection's `sender <ControlProjection.sender>`.  See `ControlMechanism_ControlSignals` for
        additional details.

      * *GATING_PROJECTION* (or *GATING*) -- this can be used when specifying an `InputPort
        <InputPort_Projection_Source_Specification>` or an `OutputPort <OutputPort_Projections>`, to create a
        default `GatingProjection` to the `Port <Port>`. If the GatingProjection's `sender <GatingProjection.sender>`
        cannot be inferred from the context in which this specification occurs, then its `initialization is deferred
        <GatingProjection_Deferred_Initialization>` until it can be determined (e.g., a `GatingMechanism` or
        `GatingSignal` is created to which it is assigned).
  ..
  * **value** -- creates a Projection of a type determined by the context of the specification, and using the
    specified value as the `value <Projection_Base.value>` of the Projection, which must be compatible with the
    `variable <Port_Base.variable>` attribute of its `receiver <Projection_Base.receiver>`.  If the Projection is a
    `MappingProjection`, the value is interpreted as a `matrix specification <MappingProjection_Matrix_Specification>`
    and assigned as the `matrix <MappingProjection.matrix>` parameter of the Projection;  it must be compatible with the
    `value <Port_Base.value>` attribute of its `sender <MappingProjection.sender>` and `variable <Port_Base.variable>`
    attribute of its `receiver <MappingProjection.receiver>`.
  ..
  * **Mechanism** -- creates a `MappingProjection` to either the `primary InputPort <InputPort_Primary>` or
    `primary OutputPort <OutputPort_Primary>`, depending on the type of Mechanism and context of the specification.
  ..
  * **Port** -- creates a `Projection` to or from the specified `Port`, depending on the type of Port and the
    context of the specification.

  .. _Projection_Specification_Dictionary:

  * **Specification dictionary** -- can contain an entry specifying the type of Projection, and/or entries
    specifying the value of parameters used to instantiate it. These should take the following form:

      * *PROJECTION_TYPE*: *<name of a Projection type>* --
        if this entry is absent, a default Projection will be created that is appropriate for the context
        (for example, a `MappingProjection` for an `InputPort`, a `LearningProjection` for the `matrix
        <MappingProjection.matrix>` parameter of a `MappingProjection`, and a `ControlProjection` for any other
        type of parameter.

      * *PROJECTION_PARAMS*: *Dict[Projection argument, argument value]* --
        the key for each entry of the dictionary must be the name of a Projection parameter, and its value the value
        of the parameter.  It can contain any of the standard parameters for instantiating a Projection (in particular
        its `sender <Projection_Sender>` and `receiver <Projection_Receiver>`, or ones specific to a particular type
        of Projection (see documentation for subclass).  If the `sender <Projection_Sender>` and/or
        `receiver <Projection_Receiver>` are not specified, their assignment and/or creation are handled in the same
        manner as described above for keyword specifications.

      COMMENT:
          WHAT ABOUT SPECIFICATION USING OutputPort/ModulatorySignal OR Mechanism? OR Matrix OR Matrix keyword
      COMMENT

      COMMENT:  ??IMPLEMENTED FOR PROJECTION PARAMS??
        Note that parameter
        values in the specification dictionary will be used to instantiate the Projection.  These can be overridden
        during execution by specifying `runtime parameters <Mechanism_Runtime_Params>` for the Projection,
        either when calling the `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>`
        method for a Mechanism directly, or where it is specified in the `pathway` of a Process.
      COMMENT

  .. _Projection_ProjectionTuple:

  * **ProjectionTuple** -- a 4-item tuple used in the context of a `Port specification <Port_Specification>` to
    create a Projection between it and another `Port <Port>`. It must have at least the first three of the following
    items in order, and can include the fourth optional item:

     * **Port specification** -- specifies the `Port <Port_Specification>` to connect with (**not** the one being
       connected; that is determined from context)

     * **weight** -- must be a value specifying the `weight <Projection_Base.weight>` of the Projection;  it can be
       `None`, in which case it is ignored, but there must be a specification present;

     * **exponent** -- must be a value specifying the `exponent <Projection_Base.exponent>` of the Projection;  it
       can be `None`, in which case it is ignored, but there must be a specification present;

     * **Projection specification** -- this is optional but, if included, msut be a `Projection specification
       <Projection_Specification>`;  it can take any of the forms of a Projection specification described above for
       any Projection subclass; it can be used to provide additional specifications for the Projection, such as its
       `matrix <MappingProjection.matrix>` if it is a `MappingProjection`.

    .. note::
       A ProjectionTuple should not be confused with a `4-item InputPort specification tuple
       <InputPort_Tuple_Specification>`, which also contains weight and exponent items.  In a ProjectionTuple, those
       items specify the weight and/or exponent assigned to the *Projection* (see `Projection_Weight_Exponent`),
       whereas in an `InputPort specification tuple <InputPort_Weights_And_Exponents>` they specify the weight
       and/or exponent of the **InputPort**.

    Any (but not all) of the items can be `None`.  If the Port specification is `None`, then there must be a
    Projection specification (used to infer the Port to be connected with).  If the Projection specification is
    `None` or absent, the Port specification cannot be `None` (as it is then used to infer the type of Projection).
    If weight and/or exponent is `None`, it is ignored.  If both the Port and Projection are specified, they must
    be compatible  (see `examples <Port_Projections_Examples>` in Port).


.. _Projection_Automatic_Creation:

*Automatic creation*
~~~~~~~~~~~~~~~~~~~~

Under some circumstances Projections are created automatically. For example, a `Composition` automatically creates
a `MappingProjection` between adjacent `ProcessingMechanisms <ProcessingMechanism>` specified in the **pathways**
argument of its constructor (if none is specified) or in its `add_linear_processing_pathway
<Composition.add_linear_processing_pathway>` method;  and, similarly, `LearningProjections <LearningProjection>` are
automatically created when a `learning pathway <Composition_Learning_Pathway>` is added to a Composition.

.. _Projection_Deferred_Initialization:

*Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~

When a Projection is created, its full initialization is `deferred <Component_Deferred_Init>` until its `sender
<Projection_Base.sender>` and `receiver <Projection_Base.receiver>` have been fully specified.  This allows a
Projection to be created before its `sender <Projection_Base.sender>` and/or `receiver <Projection_Base.receiver>` have
been created (e.g., before them in a script), by calling its constructor without specifying its **sender** or
**receiver** arguments. However, for the Projection to be operational, initialization must be completed by calling
its `_deferred_init` method.  Under most conditions this occurs automatically (e.g., when the projection is assigned
to a type of Component that expects to be the `sender <Projection_Base.sender>` or `receiver <Projection_Base.receiver>`
for that type of Projection); these conditions are described in the section on *Deferred Initialization* for each type
of Projection.  Otherwise, the  Projection's `_deferred_init` method must be called explicitly, once the missing
attribute assignments have been made.


.. _Projection_Structure:

Structure
---------

In addition to its `function <Projection_Base.function>`, a Projection has two primary attributes: a `sender
<Projection_Base.sender>` and `receiver <Projection_Base.receiver>`.  The types of `Port(s) <Port>` that can be
assigned to these, and the attributes of those Ports to which Projections of each type are assigned, are
summarized in the following table, and described in greater detail in the subsections below.  In addition to the
Port attributes to which different types of Projections are assigned (shown in the table), all of the Projections
of a Port are listed in its `projections <Port_Base.projections>` attribute.

.. _Projection_Table:

.. table:: **Sender, Receiver and Attribute Assignments for Projection Types**
    :align: center

    +----------------------+---------------------------------------+--------------------------------------------------+
    |     Projection       |   sender                              |  receiver                                        |
    |                      |   *(attribute)*                       |  *(attribute)*                                   |
    +======================+=======================================+==================================================+
    | `MappingProjection`  | `OutputPort`                          | `InputPort`                                      |
    |                      | (`efferents <Port.efferents>`)        | (`path_afferents <Port.path_afferents>`)         |
    +----------------------+---------------------------------------+--------------------------------------------------+
    | `LearningProjection` | `LearningSignal`                      | `ParameterPort`                                  |
    |                      | (`efferents <Port.efferents>`)        | (`mod_afferents <ParameterPort.mod_afferents>`)  |
    +----------------------+---------------------------------------+--------------------------------------------------+
    | `ControlProjection`  | `ControlSignal`                       | `InputPort`, `ParameterPort` or `OutputPort`     |
    |                      | (`efferents <Port.efferents>`)        | (`mod_afferents <ParameterPort.mod_afferents>`)  |
    +----------------------+---------------------------------------+--------------------------------------------------+
    | `GatingProjection`   | `GatingSignal`                        | `InputPort` or `OutputPort`                      |
    |                      | (`efferents <Port.efferents>`)        | (`mod_afferents <Port_Base.mod_afferents>`)      |
    +----------------------+---------------------------------------+--------------------------------------------------+

.. _Projection_Sender:

*Sender*
~~~~~~~~

This must be an `OutputPort` or a `ModulatorySignal <ModulatorySignal>` (a subclass of OutputPort specialized for
`ModulatoryProjections <ModulatoryProjection>`).  The Projection is assigned to the OutputPort or ModulatorySignal's
`efferents <Port_Base.efferents>` list and, for ModulatoryProjections, to the list of ModulatorySignals specific to
the `ModulatoryMechanism <ModulatoryMechanism>` from which it projects.  The OutputPort or ModulatorySignal's `value
<OutputPort.value>` is used as the `variable <Function.variable>` for Projection's `function
<Projection_Base.function>`.

A sender can be specified as:

  * an **OutputPort** or **ModulatorySignal**, as appropriate for the Projection's type, using any of the ways for
    `specifying an OutputPort <OutputPort_Specification>`.
  ..
  * a **Mechanism**;  for a `MappingProjection`, the Mechanism's `primary OutputPort <OutputPort_Primary>` is
    assigned as the `sender <Projection_Base.sender>`; for a `ModulatoryProjection <ModulatoryProjection>`, a
    `ModulatorySignal <ModulatorySignal>` of the appropriate type is created and assigned to the Mechanism.

If the `sender <Projection_Base.sender>` is not specified and it can't be determined from the context, or an OutputPort
specification is not associated with a Mechanism that can be determined from , then the initialization of the
Projection is `deferred <Projection_Deferred_Initialization>`.

.. _Projection_Receiver:

*Receiver*
~~~~~~~~~~

The `receiver <Projection_Base.receiver>` required by a Projection depends on its type, as listed below:

    * MappingProjection: `InputPort`
    * LearningProjection: `ParameterPort` (for the `matrix <MappingProjection>` of a `MappingProjection`)
    * ControlProjection: `ParameterPort`
    * GatingProjection: `InputPort` or OutputPort`

A `MappingProjection` (as a `PathwayProjection <PathwayProjection>`) is assigned to the `path_afferents
<Port.path_afferents>` attribute of its `receiver <Projection_Base.receiver>`.  The ModulatoryProjections are assigned
to the `mod_afferents <Port.mod_afferents>` attribute of their `receiver <Projection_Base.receiver>`.

A `receiver <Projection_Base.receiver>` can be specified as:

  * an existing **Port**;
  ..
  * an existing **Mechanism** or **Projection**; which of these is permissible, and how a port is assigned to it, is
    determined by the type of Projection — see subclasses for details).
  ..
  * a **specification dictionary** (see subclasses for details).

.. _Projection_Weight_Exponent:

*Weight and Exponent*
~~~~~~~~~~~~~~~~~~~~~

Every Projection has a `weight <Projection_Base.weight>` and `exponent <Projection_Base.exponent>` attribute. These
are applied to its `value <Projection_Base.value>` before combining it with other Projections that project to the same
`Port`.  If both are specified, the `exponent <Projection_Base.exponent>` is applied before the `weight
<Projection_Base.weight>`.  These attributes determine both how the Projection's `value <Projection_Base.value>` is
combined with others to determine the `variable <Port_Base.variable>` of the Port to which they project.

.. note::
   The `weight <Projection_Base.weight>` and `exponent <Projection_Base.exponent>` attributes of a Projection are not
   the same as a Port's `weight <Port_Base.weight>` and `exponent <Port_Base.exponent>` attributes.  Also, they are
   not normalized: their aggregate effects contribute to the magnitude of the `variable <Port.variable>` to which
   they project.


*ParameterPorts and Parameters*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ParameterPorts <ParameterPort>` provide the value for each parameter of a Projection and its `function
<Mechanism_Base.function>`.  ParameterPorts and their associated parameters are handled in the same way by
Projections as they are for Mechanisms (see `Mechanism_ParameterPorts` for details).  The ParameterPorts for a
Projection are listed in its `parameter_ports <Projection_Base.parameter_ports>` attribute.


.. _Projection_Execution:

Execution
---------

A Projection cannot be executed directly.  It is executed when the `Port <Port>` to which it projects (i.e., its
`receiver <Projection_Base.receiver>`) is updated;  that occurs when the Port's owner `Mechanism <Mechanism>` is
executed. When a Projection executes, it gets the value of its `sender <Projection_Base.sender>`, assigns this as the
`variable <Projection_Base.variable>` of its `function <Projection_Base.function>`, calls the `function
<Projection_Base.function>`, and provides the result as to its `receiver <Projection_Base.receiver>`.  The `function
<Projection_Base.function>` of a Projection converts the value received from its `sender <Projection_Base.sender>` to
a form suitable as input for its `receiver <Projection_Base.receiver>`.

COMMENT:
*** ADD EXAMPLES

GET FROM Scratch Pad

for example, if a ProjectionTuple is used in the context of an
    `InputPort specification
    <InputPort_Specification>` to specify a MappingProjection to it from an `OutputPort` that is specified
    in the first item of the tuple, and a Projection specification is included in the fourth, its sender (and/or the
    sending dimensions of its `matrix <MappingProjection.matrix>` parameter) must be compatible with the specified
    OutputPort (see `examples <XXX>` below)

COMMENT


.. _Projection_Class_Reference:

Class Reference
---------------

"""
import abc
import inspect
import itertools
import warnings
from collections import namedtuple, defaultdict

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.transferfunctions import LinearMatrix
from psyneulink.core.components.functions.function import get_matrix
from psyneulink.core.components.shellclasses import Mechanism, Process_Base, Projection, Port
from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import _is_modulatory_spec
from psyneulink.core.components.ports.port import PortError
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL, EXPONENT, FUNCTION_PARAMS, GATING, GATING_PROJECTION, GATING_SIGNAL, \
    INPUT_PORT, LEARNING, LEARNING_PROJECTION, LEARNING_SIGNAL, \
    MAPPING_PROJECTION, MATRIX, MATRIX_KEYWORD_SET, MECHANISM, \
    MODEL_SPEC_ID_RECEIVER_MECH, MODEL_SPEC_ID_RECEIVER_PORT, MODEL_SPEC_ID_SENDER_MECH, MODEL_SPEC_ID_SENDER_PORT, \
    NAME, OUTPUT_PORT, OUTPUT_PORTS, PARAMS, PATHWAY, PROJECTION, PROJECTION_PARAMS, PROJECTION_SENDER, PROJECTION_TYPE, \
    RECEIVER, SENDER, STANDARD_ARGS, PORT, PORTS, WEIGHT, ADD_INPUT_PORT, ADD_OUTPUT_PORT, \
    PROJECTION_COMPONENT_CATEGORY
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.registry import register_category, remove_instance_from_registry
from psyneulink.core.globals.socket import ConnectionInfo
from psyneulink.core.globals.utilities import ContentAddressableList, is_matrix, is_numeric

__all__ = [
    'Projection_Base', 'projection_keywords', 'PROJECTION_SPEC_KEYWORDS',
    'ProjectionError', 'DuplicateProjectionError', 'ProjectionRegistry',
    'kpProjectionTimeScaleLogEntry'
]

ProjectionRegistry = {}

kpProjectionTimeScaleLogEntry = "Projection TimeScale"

projection_keywords = set()

PROJECTION_ARGS = {PROJECTION_TYPE, SENDER, RECEIVER, WEIGHT, EXPONENT} | STANDARD_ARGS

PROJECTION_SPEC_KEYWORDS = {PATHWAY: MAPPING_PROJECTION,
                            LEARNING: LEARNING_PROJECTION,
                            LEARNING_SIGNAL: LEARNING_PROJECTION,
                            LEARNING_PROJECTION: LEARNING_PROJECTION,
                            CONTROL: CONTROL_PROJECTION,
                            CONTROL_SIGNAL: CONTROL_PROJECTION,
                            CONTROL_PROJECTION: CONTROL_PROJECTION,
                            GATING: GATING_PROJECTION,
                            GATING_SIGNAL: GATING_PROJECTION,
                            GATING_PROJECTION: GATING_PROJECTION
                            }

def projection_param_keyword_mapping():
    """Maps Projection type (key) to Projection parameter keywords (value) used for runtime_params specification
    Projection type is one specified in its componentType attribute, and registered in ProjectionRegistry
    """
    return {k: (k[:k.find('PROJECTION') - 9] + '_' + k[k.find('PROJECTION') - 9:]).upper() + '_PARAMS'
            for k in list(ProjectionRegistry.keys())}

def projection_param_keywords():
    return set(projection_param_keyword_mapping().values())


ProjectionTuple = namedtuple("ProjectionTuple", "port, weight, exponent, projection")


class ProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

class DuplicateProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# Projection factory method:
# def projection(name=NotImplemented, params=NotImplemented, context=None):
#         """Instantiates default or specified subclass of Projection
#
#         If called w/o arguments or 1st argument=NotImplemented, instantiates default subclass (ParameterPort)
#         If called with a name string:
#             - if registered in ProjectionRegistry class dictionary as name of a subclass, instantiates that class
#             - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
#         If a params dictionary is included, it is passed to the subclass
#
#         :param name:
#         :param param_defaults:
#         :return:
#         """
#
#         # Call to instantiate a particular subclass, so look up in MechanismRegistry
#         if name in ProjectionRegistry:
#             return ProjectionRegistry[name].mechanismSubclass(params)
#         # Name is not in MechanismRegistry or is not provided, so instantiate default subclass
#         else:
#             # from Components.Defaults import DefaultProjection
#             return DefaultProjection(name, params)
#

class Projection_Base(Projection):
    """
    Projection_Base(           \
        sender=None,           \
        function=LinearMatrix, \
        receiver=None          \
        )

    Base class for all Projections.

    The arguments below can be used in the constructor for any subclass of Mechanism.
    See `Component <Component_Class_Reference>` and subclasses for additional arguments and attributes.

    .. note::
       Projection is an abstract class and should *never* be instantiated by a direct call to its constructor.
       It should be created by calling the constructor for a subclass` or by using any of the other methods for
       `specifying a Projection <Projection_Specification>`.

    COMMENT:
    Gotchas
    -------
        When referring to the Mechanism that is a Projection's sender or receiver Mechanism, must add ".owner"

    ProjectionRegistry
    ------------------
        All Projections are registered in ProjectionRegistry, which maintains a dict for each subclass,
        a count for all instances of that type, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    sender : OutputPort or Mechanism : default None
        specifies the source of the Projection's input. If a `Mechanism <Mechanism>` is specified, its
        `primary OutputPort <OutputPort_Primary>` is used. If it is not specified, it is assigned in
        the context in which the Projection is used, or its initialization will be `deferred
        <Projection_Deferred_Initialization>`.

    function : TransferFunction : default LinearMatrix
        specifies function used to convey (and potentially convert) `value <Port_Base.value>` of `sender
        <Projection_Base.sender>` `Port` to `variable <Port_Base.variable>` of `receiver <Projection_Base.receiver>`
        Port.

    receiver: InputPort or Mechanism : default None
        specifies the destination of the Projection's output.  If a `Mechanism <Mechanism>` is specified, its
        `primary InputPort <InputPort_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used, or its initialization will be `deferred
        <Projection_Deferred_Initialization>`.


    Attributes
    ----------

    variable : value
        input to Projection, received from `value <OutputPort.value>` of `sender <Projection_Base.sender>`.

    sender : Port
        Port from which Projection receives its input (see `Projection_Sender` for additional information).

    receiver : Port
        Port to which Projection sends its output  (see `Projection_Receiver` for additional information)

    function :  TransferFunction
        conveys (and potentially converts) `variable <Projection_Base.variable>` to `value <Projection_Base.value>`.

    value : value
        output of Projection, transmitted to variable of function of its `receiver <Projection_Base.receiver>`.

    parameter_ports : ContentAddressableList[str, ParameterPort]
        a read-only list of the Projection's `ParameterPorts <Mechanism_ParameterPorts>`, one for each of its
        `modulable parameters <ParameterPort_Modulable_Parameters>`, including those of its `function
        <Projection_Base.function>`.  The value of the parameters of the Projection and its `function
        <Projection_Base.function>` are also accessible as (and can be modified using) attributes of the Projection,
        in the same manner as they can for a `Mechanism <Mechanism_ParameterPorts>`).

    weight : number
       multiplies the `value <Projection_Base.value>` of the Projection after applying the `exponent
       <Projection_Base.exponent>`, and before combining with any other Projections that project to the same `Port`
       to determine that Port's `variable <Port_Base.variable>` (see `Projection_Weight_Exponent` for details).

    exponent : number
        exponentiates the `value <Projection_Base.value>` of the Projection, before applying `weight
        <Projection_Base.weight>`, and before combining it with any other Projections that project to the same `Port`
        to determine that Port's `variable <Port_Base.variable>` (see `Projection_Weight_Exponent` for details).

    name : str
        the name of the Projection. If the Projection's `initialization has been deferred
        <Projection_Deferred_Initialization>`, it is assigned a temporary name (indicating its deferred initialization
        status) until initialization is completed, at which time it is assigned its designated name.  If that is the
        name of an existing Projection, it is appended with an indexed suffix, incremented for each Projection with the
        same base name (see `Registry_Naming`). If the name is not  specified in the **name** argument of its
        constructor, a default name is assigned by the subclass (see subclass for details)

    """

    color = 0

    componentCategory = PROJECTION_COMPONENT_CATEGORY
    className = componentCategory
    suffix = " " + className

    class Parameters(Projection.Parameters):
        """
            Attributes
            ----------

                exponent
                    see `exponent <Projection_Base.exponent>`

                    :default value: None
                    :type:

                function
                    see `function <Projection_Base.function>`

                    :default value: `LinearMatrix`
                    :type: `Function`

                weight
                    see `weight <Projection_Base.weight>`

                    :default value: None
                    :type:
        """
        weight = Parameter(None, modulable=True)
        exponent = Parameter(None, modulable=True)
        function = Parameter(LinearMatrix, stateful=False, loggable=False)

    registry = ProjectionRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    @abc.abstractmethod
    def __init__(self,
                 receiver,
                 sender=None,
                 weight=None,
                 exponent=None,
                 function=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None,
                 **kwargs
                 ):
        """Assign sender, receiver, and execute method and register Mechanism with ProjectionRegistry

        This is an abstract class, and can only be called from a subclass;
           it must be called by the subclass with a context value

        # DOCUMENT:  MOVE TO ABOVE, UNDER INSTANTIATION
        Initialization arguments:
            - sender (Mechanism, Port or dict):
                specifies source of input to Projection (default: senderDefault)
            - receiver (Mechanism, Port or dict)
                 destination of Projection (default: none)
            - params (dict) - dictionary of Projection params:
                + FUNCTION:<method>
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
            - prefs (PreferenceSet or specification dict):
                 if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
                 dict entries must have a preference keyPath as key, and a PreferenceEntry or setting as their value
                 (see Description under PreferenceSet for details)
            - context (str): must be a reference to a subclass, or an exception will be raised

        NOTES:
        * Receiver is required, since can't instantiate a Projection without a receiving Port
        * If sender and/or receiver is a Mechanism, the appropriate Port is inferred as follows:
            MappingProjection:
                sender = <Mechanism>.output_port
                receiver = <Mechanism>.input_port
            ControlProjection:
                sender = <Mechanism>.output_port
                receiver = <Mechanism>.<param> IF AND ONLY IF there is a single one
                            that is a ParameterPort;  otherwise, an exception is raised
        * _instantiate_sender, _instantiate_receiver must be called before _instantiate_function:
            - _validate_params must be called before _instantiate_sender, as it validates PROJECTION_SENDER
            - instantatiate_sender may alter self.defaults.variable, so it must be called before _validate_function
            - instantatiate_receiver must be called before _validate_function,
                 as the latter evaluates receiver.value to determine whether to use self.function or FUNCTION
        * If variable is incompatible with sender's output, it is set to match that and revalidated (_instantiate_sender)
        * if FUNCTION is provided but its output is incompatible with receiver value, self.function is tried
        * registers Projection with ProjectionRegistry

        :param sender: (Port or dict)
        :param receiver: (Port or dict)
        :param param_defaults: (dict)
        :param name: (str)
        :param context: (str)
        :return: None
        """
        from psyneulink.core.components.ports.parameterport import ParameterPort
        from psyneulink.core.components.ports.port import Port_Base

        if self.initialization_status == ContextFlags.DEFERRED_INIT:
            self._assign_deferred_init_name(name)
            self._store_deferred_init_args(**locals())
            return

        self.receiver = receiver

         # Register with ProjectionRegistry or create one
        register_category(entry=self,
                          base_class=Projection_Base,
                          name=name,
                          registry=ProjectionRegistry,
                          )

        # Create projection's _portRegistry and ParameterPort entry
        self._portRegistry = {}

        register_category(entry=ParameterPort,
                          base_class=Port_Base,
                          registry=self._portRegistry,
                          )

        self._instantiate_sender(sender, context=context)

        # FIX: ADD _validate_variable, THAT CHECKS FOR SENDER?
        # FIX: NEED TO KNOW HERE IF SENDER IS SPECIFIED AS A MECHANISM OR PORT
        try:
            # this should become _default_value when that is fully implemented
            variable = self.sender.defaults.value
        except AttributeError:
            if receiver.prefs.verbosePref:
                warnings.warn("Unable to get value of sender ({0}) for {1};  will assign default ({2})".
                              format(self.sender, self.name, self.class_defaults.variable))
            variable = None

        # Assume that if receiver was specified as a Mechanism, it should be assigned to its (primary) InputPort
        # MODIFIED 11/1/17 CW: Added " hasattr(self, "prefs") and" in order to avoid errors. Otherwise, this was being
        # called and yielding an error: " AttributeError: 'MappingProjection' object has no attribute '_prefs' "

        if isinstance(self.receiver, Mechanism):
            if (len(self.receiver.input_ports) > 1 and hasattr(self, 'prefs') and
                    (self.prefs.verbosePref or self.receiver.prefs.verbosePref)):
                print("{0} has more than one InputPort; {1} has been assigned to the first one".
                      format(self.receiver.owner.name, self.name))
            self.receiver = self.receiver.input_port

        if hasattr(self.receiver, "afferents_info"):
            if self not in self.receiver.afferents_info:
                self.receiver.afferents_info[self] = ConnectionInfo()

       # Validate variable, function and params
        # Note: pass name of Projection (to override assignment of componentName in super.__init__)
        super(Projection_Base, self).__init__(
            default_variable=variable,
            function=function,
            param_defaults=params,
            weight=weight,
            exponent=exponent,
            name=self.name,
            prefs=prefs,
            **kwargs
        )

        self._assign_default_projection_name()

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate PROJECTION_SENDER and/or sender arg (current self.sender), and assign one of them as self.sender

        Check:
        - that PROJECTION_SENDER is a Mechanism or Port
        - if it is different from .projection_sender, use it
        - if it is the same or is invalid, check if sender arg was provided to __init__ and is valid
        - if sender arg is valid use it (if PROJECTION_SENDER can't be used);
        - if both were not provided, use .projection_sender
        - otherwise, if one was not provided and the other is invalid, generate error
        - when done, sender is assigned to self.sender

        Note: check here only for sender's type, NOT content (e.g., length, etc.); that is done in _instantiate_sender

        :param request_set:
        :param target_set:
        :param context:
        :return:
        """

        super(Projection, self)._validate_params(request_set, target_set, context)

        # FIX: 10/3/17 SHOULD ADD CHECK THAT RECEIVER/SENDER SOCKET SPECIFICATIONS ARE CONSISTENT WITH
        # FIX:         PROJECTION_TYPE SPECIFIED BY THE CORRESPONDING PORT TYPES

        if (PROJECTION_SENDER in target_set and
                not (target_set[PROJECTION_SENDER] in {None, self.projection_sender})):
            # If PROJECTION_SENDER is specified it will be the sender
            sender = target_set[PROJECTION_SENDER]
            sender_string = PROJECTION_SENDER
        else:
            # PROJECTION_SENDER is not specified or None, so sender argument of constructor will be the sender
            sender = self.sender
            sender_string = "\'{}\' argument".format(SENDER)
        if not ((isinstance(sender, (Mechanism, Port)) or
                 (inspect.isclass(sender) and issubclass(sender, (Mechanism, Port))))):
            raise ProjectionError("Specification of {} for {} ({}) is invalid; "
                                  "it must be a {}, {} or a class of one of these.".
                                  format(sender_string, self.name, sender,
                                         Mechanism.__name__, Port.__name__))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        self._instantiate_parameter_ports(function=function, context=context)

    def _instantiate_parameter_ports(self, function=None, context=None):

        from psyneulink.core.components.ports.parameterport import _instantiate_parameter_ports
        _instantiate_parameter_ports(owner=self, function=function, context=context)

    def _instantiate_sender(self, sender, context=None):
        """Assign self.sender to OutputPort of sender

        Assume self.sender has been assigned in _validate_params, from either sender arg or PROJECTION_SENDER
        Validate, and assign projection to sender's efferents attribute

        If self.sender is a Mechanism, re-assign it to <Mechanism>.output_port
        If self.sender is a Port class reference, validate that it is a OutputPort
        Assign projection to sender's efferents attribute
        """
        from psyneulink.core.compositions.composition import Composition
        from psyneulink.core.components.ports.outputport import OutputPort

        if not (
            isinstance(sender, (Composition, Mechanism, Port, Process_Base))
            or (inspect.isclass(sender) and issubclass(sender, (Mechanism, Port)))
        ):
            assert False, \
                f"PROGRAM ERROR: Invalid specification for {SENDER} ({sender}) of {self.name} " \
                f"(including class default: {self.projection_sender})."

        # If self.sender is specified as a Mechanism (rather than a Port),
        #     get relevant OutputPort and assign it to self.sender
        # IMPLEMENTATION NOTE: Assume that self.sender should be the primary OutputPort; if that is not the case,
        #                      self.sender should either be explicitly assigned, or handled in an override of the
        #                      method by the relevant subclass prior to calling super
        if isinstance(sender, Composition):
            sender = sender.output_CIM
        if isinstance(sender, Mechanism):
            sender = sender.output_port
        self.sender = sender

        # At this point, self.sender should be a OutputPort
        if not isinstance(self.sender, OutputPort):
            raise ProjectionError("Sender specified for {} ({}) must be a Mechanism or an OutputPort".
                                  format(self.name, self.sender))

        # Assign projection to self.sender's efferents list attribute
        # First make sure that projection is not already in efferents
        # IMPLEMENTATON NOTE:  Currently disallows *ANY* Projections with same sender and receiver
        #                      (even if they are in different Compositions)
        if self not in self.sender.efferents:
            # Then make sure there is not already a projection to its receiver
            receiver = self.receiver
            if isinstance(receiver, Composition):
                receiver = receiver.input_CIM
            if isinstance(receiver, Mechanism):
                receiver = receiver.input_port
            assert isinstance(receiver, (Port)), \
                f"Illegal receiver ({receiver}) detected in _instantiate_sender() method for {self.name}"
            dup = receiver._check_for_duplicate_projections(self)
            # If duplicate is a deferred_init Projection, delete it and use one currently being instantiated
            # IMPLEMENTATION NOTE:  this gives precedence to a Projection to a Component specified by its sender
            #                      (e.g., controller of a Composition for a ControlProjection)
            #                       over its specification in the constructor for the receiver or its owner
            # IMPLEMENTATION NOTE:  This should be removed if/when different Projections are permitted between
            #                       the same sender and receiver in different Compositions
            if dup:
                if dup.initialization_status == ContextFlags.DEFERRED_INIT:
                    del receiver.mod_afferents[receiver.mod_afferents.index(dup)]
                else:
                    raise DuplicateProjectionError(f"Attempt to assign {Projection.__name__} to {receiver.name} of "
                                                   f"{receiver.owner.name} that already has an identical "
                                                   f"{Projection.__name__}.")
            self.sender.efferents.append(self)
        else:
            raise DuplicateProjectionError(f"Attempt to assign {Projection.__name__} from {sender.name} of "
                                           f"{sender.owner.name} that already has an identical {Projection.__name__}.")

    def _instantiate_attributes_after_function(self, context=None):
        from psyneulink.core.components.ports.parameterport import _instantiate_parameter_port
        self._instantiate_receiver(context=context)
        # instantiate parameter ports from UDF custom parameters if necessary
        try:
            cfp = self.function.cust_fct_params
            udf_parameters_lacking_ports = {param_name: cfp[param_name]
                                            for param_name in cfp if param_name not in self.parameter_ports.names}

            _instantiate_parameter_port(self, FUNCTION_PARAMS,
                                        udf_parameters_lacking_ports,
                                        context=context,
                                        function=self.function)
        except AttributeError:
            pass

        super()._instantiate_attributes_after_function(context=context)

    def _instantiate_receiver(self, context=None):
        """Call receiver's owner to add projection to its afferents list

        Notes:
        * Assume that subclasses implement this method in which they:
          - test whether self.receiver is a Mechanism and, if so, replace with Port appropriate for projection
          - calls this method (as super) to assign projection to the Mechanism
        * Constraint that self.value is compatible with receiver.input_port.value
            is evaluated and enforced in _instantiate_function, since that may need to be modified (see below)
        * Verification that projection has not already been assigned to receiver is handled by _add_projection_to;
            if it has, a warning is issued and the assignment request is ignored

        :param context: (str)
        :return:
        """
        # IMPLEMENTATION NOTE: since projection is added using Mechanism.add_projection(projection, port) method,
        #                      could add port specification as arg here, and pass through to add_projection()
        #                      to request a particular port
        # IMPLEMENTATION NOTE: should check that projection isn't already received by receivers

        if isinstance(self.receiver, Port):
            _add_projection_to(receiver=self.receiver.owner,
                               port=self.receiver,
                               projection_spec=self,
                               context=context)

        # This should be handled by implementation of _instantiate_receiver by projection's subclass
        elif isinstance(self.receiver, Mechanism):
            raise ProjectionError("PROGRAM ERROR: receiver for {0} was specified as a Mechanism ({1});"
                                  "this should have been handled by _instantiate_receiver for {2}".
                                  format(self.name, self.receiver.name, self.__class__.__name__))

        else:
            raise ProjectionError("Unrecognized receiver specification ({0}) for {1}".format(self.receiver, self.name))

    def _update_parameter_ports(self, runtime_params=None, context=None):
        for port in self._parameter_ports:
            port_Name = port.name
            port._update(params=runtime_params, context=context)

            # Assign version of ParameterPort.value matched to type of template
            #    to runtime param
            # FYI (7/18/17 CW) : in addition to the params and attribute being set, the port's variable is ALSO being
            # set by the statement below. For example, if port_Name is 'matrix', the statement below sets
            # params['matrix'] to port.value, calls setattr(port.owner, 'matrix', port.value), which sets the
            # 'matrix' ParameterPort's variable to ALSO be equal to port.value! If this is unintended, please change.
            value = port.parameters.value._get(context)
            getattr(self.parameters, port_Name)._set(value, context)
            # manual setting of previous value to matrix value (happens in above param['matrix'] setting
            if port_Name == MATRIX:
                port.function.parameters.previous_value._set(value, context)

    def add_to(self, receiver, port, context=None):
        _add_projection_to(receiver=receiver, port=port, projection_spec=self, context=context)

    def _execute(self, variable=None, context=None, runtime_params=None):
        if variable is None:
            variable = self.sender.parameters.value._get(context)

        value = super()._execute(
            variable=variable,
            context=context,
            runtime_params=runtime_params,

        )
        return value

    def _activate_for_compositions(self, composition):
        try:
            self.receiver.afferents_info[self].add_composition(composition)
        except KeyError:
            self.receiver.afferents_info[self] = ConnectionInfo(compositions=composition)

        try:
            if self not in composition.projections:
                composition._add_projection(self)
        except AttributeError:
            # composition may be ALL or None, in this case we don't need to add
            pass

    def _activate_for_all_compositions(self):
        self._activate_for_compositions(ConnectionInfo.ALL)

    def _delete_projection(projection, context=None):
        """Delete Projection, its entries in receiver and sender Ports, and in ProjectionRegistry"""
        projection.sender._remove_projection_from_port(projection)
        projection.receiver._remove_projection_to_port(projection)
        remove_instance_from_registry(ProjectionRegistry, projection.__class__.__name__,
                                      component=projection)

    # FIX: 10/3/17 - replace with @property on Projection for receiver and sender
    @property
    def socket_assignments(self):

        if self.initialization_status == ContextFlags.DEFERRED_INIT:
            sender = self._init_args[SENDER]
            receiver = self._init_args[RECEIVER]
        else:
            sender = self.sender
            receiver = self.receiver

        return {SENDER:sender,
                RECEIVER:receiver}

    def _projection_added(self, projection, context=None):
        """Stub that can be overidden by subclasses that need to know when a projection is added to the Projection"""
        pass

    def _assign_default_name(self, **kwargs):
        self._assign_default_projection_name(**kwargs)

    def _assign_default_projection_name(self, port=None, sender_name=None, receiver_name=None):
        raise ProjectionError("PROGRAM ERROR: {} must implement _assign_default_projection_name().".
                              format(self.__class__.__name__))

    @property
    def parameter_ports(self):
        """Read-only access to _parameter_ports"""
        return self._parameter_ports

    # Provide invocation wrapper
    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        mf_state = pnlvm.helpers.get_state_ptr(builder, self, state, self.parameters.function.name)
        mf_params = pnlvm.helpers.get_param_ptr(builder, self, params, self.parameters.function.name)
        main_function = ctx.import_llvm_function(self.function)
        builder.call(main_function, [mf_params, mf_state, arg_in, arg_out])

        return builder

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self.parameter_ports,
        ))

    @property
    def _model_spec_parameter_blacklist(self):
        """
            A set of Parameter names that should not be added to the generated
            constructor string
        """
        return super()._model_spec_parameter_blacklist.union(
            {'variable'}
        )

    @property
    def _dict_summary(self):
        # these may occur during deferred init
        if not isinstance(self.sender, type):
            sender_name = self.sender.name
            sender_mech = self.sender.owner.name
        else:
            sender_name = None
            sender_mech = None

        if not isinstance(self.receiver, type):
            receiver_name = self.receiver.name
            receiver_mech = self.receiver.owner.name
        else:
            receiver_name = None
            receiver_mech = None

        socket_dict = {
            MODEL_SPEC_ID_SENDER_PORT: sender_name,
            MODEL_SPEC_ID_RECEIVER_PORT: receiver_name,
            MODEL_SPEC_ID_SENDER_MECH: sender_mech,
            MODEL_SPEC_ID_RECEIVER_MECH: receiver_mech
        }

        return {
            **super()._dict_summary,
            **socket_dict
        }

@tc.typecheck
def _is_projection_spec(spec, proj_type:tc.optional(type)=None, include_matrix_spec=True):
    """Evaluate whether spec is a valid Projection specification

    Return `True` if spec is any of the following:
    + Projection object, and of specified type (if proj_type is specified)
    + Projection class (or keyword string constant for one), and of specified type (if proj_type is specified)
    + 2-item tuple of which the second is a projection_spec (checked recursively with this method):
    + specification dict containing:
        + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
    + valid matrix specification (if include_matrix_spec is set to `True`)
    + port

    Otherwise, return :keyword:`False`
    """

    if isinstance(spec, Projection):
        if proj_type is None or isinstance(spec, proj_type):
                return True
        else:
            return False
    if isinstance(spec, Port):
        # FIX: CHECK PORT AGAIN ALLOWABLE PORTS IF type IS SPECIFIED
        return True
    # # MODIFIED 11/29/17 NEW:
    # if isinstance(spec, Mechanism):
    #     if proj_type is None:
    #     # FIX: CHECK PORT AGAIN ALLOWABLE PORTS IF type IS SPECIFIED
    #         return True
    # MODIFIED 11/29/17 END
    if inspect.isclass(spec):
        if issubclass(spec, Projection):
            if proj_type is None or issubclass(spec, proj_type):
                return True
            else:
                return False
        if issubclass(spec, Port):
            # FIX: CHECK PORT AGAIN ALLOWABLE PORTS IF type IS SPECIFIED
            return True
    # # MODIFIED 11/29/17 NEW:
        # if issubclass(spec, Mechanism):
        #     # FIX: CHECK PORT AGAIN ALLOWABLE PORTS IF type IS SPECIFIED
        #     return True
    # MODIFIED 11/29/17 END
    if isinstance(spec, dict) and any(key in spec for key in {PROJECTION_TYPE, SENDER, RECEIVER, MATRIX}):
        # FIX: CHECK PORT AGAIN ALLOWABLE PORTS IF type IS SPECIFIED
        return True
    if isinstance(spec, str) and spec in PROJECTION_SPEC_KEYWORDS:
        # FIX: CHECK PORT AGAIN ALLOWABLE PORTS IF type IS SPECIFIED
        return True
    if include_matrix_spec:
        if isinstance(spec, str) and spec in MATRIX_KEYWORD_SET:
            return True
        if get_matrix(spec) is not None:
            return True
    if isinstance(spec, tuple) and len(spec) == 2:
        # Call recursively on first item, which should be a standard projection spec
        if _is_projection_spec(spec[0], proj_type=proj_type, include_matrix_spec=include_matrix_spec):
            if spec[1] is not None:
                # IMPLEMENTATION NOTE: keywords must be used to refer to subclass, to avoid import loop
                if _is_projection_subclass(spec[1], MAPPING_PROJECTION):
                    return True
                if _is_modulatory_spec(spec[1]):
                    return True
        # if _is_projection_spec(spec[1], proj_type=proj_type, include_matrix_spec=include_matrix_spec):
        #         # IMPLEMENTATION NOTE: keywords must be used to refer to subclass, to avoid import loop
        #     if is_numeric(spec[0]):
        #         # if _is_projection_subclass(spec[1], MAPPING_PROJECTION):
        #         #     return True
        #         if _is_modulatory_spec(spec[1]):
        #             return True

    return False


def _is_projection_subclass(spec, keyword):
    """Evaluate whether spec is a valid specification of type

    keyword must specify a class registered in ProjectionRegistry

    Return true if spec ==
    + keyword
    + subclass of Projection associated with keyword (from ProjectionRegistry)
    + instance of the subclass
    + specification dict for instance of the subclass:
        keyword is a keyword for an entry in the spec dict
        keyword[spec] is a legal specification for the subclass

    Otherwise, return :keyword:`False`
    """
    if spec is keyword:
        return True
    # Get projection subclass specified by keyword
    try:
        proj_type = ProjectionRegistry[keyword].subclass
    except KeyError:
        pass
    else:
        # Check if spec is either the name of the subclass or an instance of it
        if inspect.isclass(spec) and issubclass(spec, proj_type):
            return True
        if isinstance(spec, proj_type):
            return True
    # spec is a specification dict for an instance of the projection subclass
    if isinstance(spec, dict) and keyword in spec:
        # Recursive call to determine that the entry of specification dict is a legal spec for the projection subclass
        if _is_projection_subclass(spec[keyword], keyword):
            return True
    return False


def _parse_projection_spec(projection_spec,
                           owner = None,       # Used only for error message
                           port_type = None,  # Used only for default assignment
                           # socket=None,
                           **kwargs):
    """Return either Projection object or Projection specification dict for projection_spec

    All keys in kwargs must be from PROJECTION_ARGS

    If projection_spec is or resolves to a Projection object, returns Projection object.
    Otherwise, return Projection specification dictionary using any arguments provided as defaults
    """

    bad_arg = next((key for key in kwargs if key not in PROJECTION_ARGS), None)
    if bad_arg:
        raise ProjectionError("Illegal argument in call to _parse_port_spec: {}".format(bad_arg))

    proj_spec_dict = defaultdict(lambda :None)
    proj_spec_dict.update(kwargs)

    # Projection object
    if isinstance(projection_spec, Projection):
        projection = projection_spec
        # FIX: NOT SURE WHICH TO GIVE PRECEDENCE: SPEC IN ProjectionTuple OR INSTANTIATED Projection:
        if ((proj_spec_dict[WEIGHT] is not None and projection.weight is not None) or
            (proj_spec_dict[EXPONENT] is not None and projection.exponent is not None)):
            raise ProjectionError("PROGRAM ERROR: Conflict in weight and/or exponent specs "
                                  "between Projection and ProjectionTuple")
        if projection.initialization_status == ContextFlags.DEFERRED_INIT:
            projection._init_args[NAME] = proj_spec_dict[NAME] or projection._init_args[NAME]
        else:
            projection.name = proj_spec_dict[NAME] or projection.name

        return projection

    # Projection class
    elif inspect.isclass(projection_spec) and issubclass(projection_spec, Projection):
        proj_spec_dict[PROJECTION_TYPE] = projection_spec

    # Matrix
    elif is_matrix(projection_spec):
        is_matrix(projection_spec)
        proj_spec_dict[MATRIX] = projection_spec

    # Projection keyword
    elif isinstance(projection_spec, str):
        proj_spec_dict[PROJECTION_TYPE] = _parse_projection_keyword(projection_spec)

    # Port object or class
    elif (isinstance(projection_spec, Port)
          or (isinstance(projection_spec, type) and issubclass(projection_spec, Port))):
        proj_spec_dict[PROJECTION_TYPE] = projection_spec.projection_type
        port_type = projection_spec.__class__

    # Mechanism object or class
    elif (isinstance(projection_spec, Mechanism)
          or (isinstance(projection_spec, type) and issubclass(projection_spec, Mechanism))):
        proj_spec_dict[PROJECTION_TYPE] = projection_spec.outputPortTypes.projection_type

    # Dict
    elif isinstance(projection_spec, dict):

        proj_spec_dict = projection_spec

        # Get projection params from specification dict
        if PROJECTION_PARAMS in proj_spec_dict:
            proj_spec_dict[PARAMS].update = proj_spec_dict[PROJECTION_PARAMS]
            # projection_spec[PARAMS].update(projection_params)
            assert False, "PROJECTION_PARAMS ({}) passed in spec dict in ProjectionTuple for {}.".\
                           format(proj_spec_dict[PROJECTION_PARAMS], projection_spec, proj_spec_dict[NAME])

    # None
    if not proj_spec_dict[PROJECTION_TYPE]:
        # Assign default type
        proj_spec_dict[PROJECTION_TYPE] = port_type.projection_type

        # prefs is not always created when this is called, so check
        try:
            owner.prefs
            has_prefs = True
        except AttributeError:
            has_prefs = False

        if has_prefs and owner.prefs.verbosePref:
            warnings.warn("Unrecognized specification ({}) for a Projection for {} of {}; "
                          "default {} has been assigned".
                          format(projection_spec,
                                 port_type.__class__.__name__,
                                 owner.name,
                                 proj_spec_dict[PROJECTION_TYPE]))
    return proj_spec_dict


def _parse_projection_keyword(projection_spec:str):
    """Takes keyword (str) and returns corresponding Projection class
    """
    # get class for keyword in registry
    try:
        projection_type = ProjectionRegistry[PROJECTION_SPEC_KEYWORDS[projection_spec]].subclass
    except KeyError:
        # projection_spec was not a recognized key
        raise ProjectionError("{} is not a recognized {} keyword".format(projection_spec, Projection.__name__))
    # projection_spec was legitimate keyword
    else:
        return projection_type


def _parse_connection_specs(connectee_port_type,
                            owner,
                            connections):
    """Parse specification(s) for Ports to/from which the connectee_port_type should be connected

    TERMINOLOGY NOTE:
        "CONNECTION" is used instead of "PROJECTION" because:
            - the method abstracts over type and direction of Projection, so it is ambiguous whether
                the projection involved is to or from connectee_port_type; however, can always say it "connects with"
            - specification is not always (in fact, usually is not) in the form of a Projection; usually it is a
                Mechanism or Port to/from which the connectee_port_type should send/receive the Projection

    Connection attributes declared for each type (subclass) of Port that are used here:
        connectsWith : Port
           - specifies the type (subclass) of Port with which the connectee_port_type should be connected
        connectsWithAttribute : str
           - specifies the name of the attribute of the Mechanism that holds the ports of the connectsWith's type
        projectionSocket : [SENDER or RECEIVER]
           - specifies for this method whether to use a Projection's sender or receiver for the connection
        modulators : ModulatorySignal
           -  class of ModulatorySignal that can send ModulatoryProjection to the connectee_port_type

    This method deals with connection specifications that are made in one of the following places/ways:
        - *PROJECTIONS* entry of a Port specification dict;
        - last item of a Port specification tuple.

    In both cases, the connection specification can be a single (stand-alone) item or a list of them.

    Projection(s) in connection(s) can be specified in any of the ways a Projection can be specified;
        * Mechanism specifications are resolved to a primary InputPort or OutputPort, as appropriate
        * Port specifications are assumed to be for connect_with Port,
            and checked for compatibilty of assignment (using projection_socket)
        * keyword specifications are resolved to corresponding Projection class
        * Class assignments are checked for compatiblity with connectee_port_type and connect_with Port

    Each connection specification can, itself, be one of the following:
        * Port object or class;
        * Mechanism object or class - primary Port is used, if applicable, otherwise an exception is generated;
        * dict - must have the first and can have any of the additional entries below:
            *PORT*:<port_spec> - required; must resolve to an instantiated port;  can use any of the following:
                                       Port - the Port is used;
                                       Mechanism - primary Port will be used if appropriate,
                                                   otherwise generates an exception;
                                       {Mechanism:port_spec or [port_spec<, port_spec...>]} -
                                                   each port_spec must be for an instantiated Port of the Mechanism,
                                                   referenced by its name or in a CONNECTION specification that uses
                                                   its name (or, for completeness, the Port itself);
                                                   _parse_connections() is called recursively for each port_spec
                                                   (first replacing the name with the actual port);
                                                   and returns a list of ProjectionTuples; any weights, exponents,
                                                   or projections assigned in those tuples are left;  otherwise, any
                                                   values in the entries of the outer dict (below) are assigned;
                                                   note:  the dictionary can have multiple Mechanism entries
                                                          (which permits the same defaults to be assigned to all the
                                                          Ports for all of the Mechanisms)
                                                          or they can be assigned each to their own dictionary
                                                          (which permits different defaults to be assigned to the
                                                          Ports for each Mechanism);
            *WEIGHT*:<int> - optional; specifies weight given to projection by receiving InputPort
            *EXPONENT:<int> - optional; specifies weight given to projection by receiving InputPort
            *PROJECTION*:<projection_spec> - optional; specifies projection (instantiated or matrix) for connection
                                             default is PROJECTION_TYPE specified for PORT
        * tuple or list of tuples: (specification requirements same as for dict above);  each must be:
            (port_spec, projection_spec) or
            (port_spec, weight, exponent, projection_spec)

    Returns list of ProjectionTuples, each of which specifies:
        - the port to be connected with
        - weight and exponent for that connection (assigned to the projection)
        - projection specification

    """

    from psyneulink.core.components.ports.port import _get_port_for_socket
    from psyneulink.core.components.ports.port import PortRegistry
    from psyneulink.core.components.ports.inputport import InputPort
    from psyneulink.core.components.ports.outputport import OutputPort
    from psyneulink.core.components.ports.parameterport import ParameterPort
    from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
    from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import _is_control_spec
    from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import _is_gating_spec

    if not inspect.isclass(connectee_port_type):
        raise ProjectionError("Called for {} with \'connectee_port_type\' arg ({}) that is not a class".
                         format(owner.name, connectee_port_type))

    # Get connection attributes for connectee
    connects_with = [PortRegistry[name].subclass for name in connectee_port_type.connectsWith]
    connect_with_attr = connectee_port_type.connectsWithAttribute
    projection_socket = connectee_port_type.projectionSocket
    modulators = [PortRegistry[name].subclass for name in connectee_port_type.modulators]

    DEFAULT_WEIGHT = None
    DEFAULT_EXPONENT = None
    DEFAULT_PROJECTION = None

    # Convert to list for subsequent processing
    if isinstance(connections, set):
        # if owner.verbosePref:
        #     warnings.warn("Connection specification for {} of {} was a set ({});"
        #                   "it was converted to a list, but the order of {} assignments is not "
        #                   "predictable".format(connectee_port_type, owner.name,
        #                                        connections, Projection.__name__))
        # connections = list(connections)
        raise ProjectionError("Connection specification for {} of {} is a set ({}); it should be a list.".
                              format(connectee_port_type.__name__, owner.name, connections, Projection.__name__))

    elif not isinstance(connections, list):
        connections = [connections]
    connect_with_ports = []

    for connection in connections:

        # If a Mechanism, Port, or Port type is used to specify the connection on its own (i.e., w/o dict or tuple)
        #     put in ProjectionTuple as both Port spec and Projection spec (to get Projection for that Port)
        #     along with defaults for weight and exponent, and call _parse_connection_specs recursively
        #     to validate the port spec and append ProjectionTuple to connect_with_ports
        if isinstance(connection, (Mechanism, Port, type)):
            # FIX: 10/3/17 - REPLACE THIS (AND ELSEWHERE) WITH ProjectionTuple THAT HAS BOTH SENDER AND RECEIVER
            # FIX: 11/28/17 - HACKS TO HANDLE PROJECTION FROM GatingSignal TO InputPort or OutputPort
            # FIX:            AND PROJECTION FROM ControlSignal to ParameterPort
            # # If it is a ModulatoryMechanism specification, get its ModulatorySignal class
            # # (so it is recognized by _is_projection_spec below (Mechanisms are not for secondary reasons)
            # if isinstance(connection, type) and issubclass(connection, ModulatoryMechanism_Base):
            #     connection = connection.outputPortTypes
            if ((isinstance(connectee_port_type, (InputPort, OutputPort, ParameterPort))
                 or isinstance(connectee_port_type, type)
                and issubclass(connectee_port_type, (InputPort, OutputPort, ParameterPort)))
                and _is_modulatory_spec(connection)):
                # Convert ModulatoryMechanism spec to corresponding ModulatorySignal spec
                if isinstance(connection, type) and issubclass(connection, ModulatoryMechanism_Base):
                    # If the connection supports multiple outputPortTypes,
                    #    get the one compatible with the current connectee:
                    output_port_types = connection.outputPortTypes
                    if not isinstance(output_port_types, list):
                        output_port_types = [output_port_types]
                    output_port_type = [o for o in output_port_types if o.__name__ in
                                          connectee_port_type.connectsWith]
                    assert len(output_port_type)==1, \
                        f"PROGRAM ERROR:  More than one {OutputPort.__name__} type found for {connection}  " \
                            f"({output_port_types}) that can be assigned a modulatory {Projection.__name__} " \
                            f"to {connectee_port_type.__name__} of {owner.name}"
                    connection = output_port_type[0]
                elif isinstance(connection, ModulatoryMechanism_Base):
                    connection = connection.output_port

                projection_spec = connection

            else:
                projection_spec = connectee_port_type

            projection_tuple = (connection, DEFAULT_WEIGHT, DEFAULT_EXPONENT, projection_spec)
            connect_with_ports.extend(_parse_connection_specs(connectee_port_type, owner, projection_tuple))

        # If a Projection specification is used to specify the connection:
        #  assign the Projection specification to the projection_specification item of the tuple,
        #  but also leave it is as the connection specification (it will get resolved to a Port reference when the
        #    tuple is created in the recursive call to _parse_connection_specs below).
        elif _is_projection_spec(connection, include_matrix_spec=False):
            projection_spec = connection
            projection_tuple = (connection, DEFAULT_WEIGHT, DEFAULT_EXPONENT, projection_spec)
            connect_with_ports.extend(_parse_connection_specs(connectee_port_type, owner, projection_tuple))

        # Dict of one or more Mechanism specifications, used to specify individual Ports of (each) Mechanism;
        #   convert all entries to tuples and call _parse_connection_specs recursively to generate ProjectionTuples;
        #   main purpose of this is to resolve any str references to name of port (using context of owner Mechanism)
        elif isinstance(connection, dict):

            # Check that dict has at least one entry with a Mechanism as the key
            if (not any(isinstance(spec, Mechanism) for spec in connection) and
                    not any(spec == PORTS for spec in connection)):
                raise ProjectionError("There are no {}s or {}s in the list ({}) specifying {}s for an {} of {}".
                                 format(Mechanism.__name__, Port.__name__, connection, Projection.__name__,
                                        connectee_port_type.__name__, owner.name))

            # Add default WEIGHT, EXPONENT, and/or PROJECTION specification for any that are not aleady in the dict
            #    (used as the default values for all the Ports of all Mechanisms specified for this dict;
            #    can use different dicts to implement different sets of defaults for the Ports of diff Mechanisms)
            if WEIGHT not in connection:
                connection[WEIGHT] = DEFAULT_WEIGHT
            if EXPONENT not in connection:
                connection[EXPONENT] = DEFAULT_EXPONENT
            if PROJECTION not in connection:
                connection[PROJECTION] = DEFAULT_PROJECTION

            # Now process each entry that has *PORTS* or a Mechanism as its key
            for key, port_connect_specs in connection.items():

                # Convert port_connect_specs to a list for subsequent processing
                if not isinstance(port_connect_specs, list):
                    port_connect_specs = [port_connect_specs]

                for port_connect_spec in port_connect_specs:

                    # Port, str (name) or Projection specification
                    if isinstance(port_connect_spec, (Port, str, _is_projection_spec)):

                        # If port_connection_spec is a string (name), it has to be in a Mechanism entry
                        if isinstance(port_connect_spec, str) and isinstance(key, Mechanism):
                            mech = key
                        else:
                            raise ProjectionError("{} specified by name ({}) is not in a {} entry".
                                                  format(Port.__name__, port_connect_spec, Mechanism.__name__))

                        # Call _get_port_for_socket to parse if it is a str,
                        #    and in either case to make sure it belongs to mech
                        port = _get_port_for_socket(owner=owner,
                                                      port_spec=port_connect_spec,
                                                      port_types=connect_with_attr,
                                                      mech=mech,
                                                      projection_socket=projection_socket)
                        if isinstance(port, list):
                            assert False, 'Got list of allowable ports for {} as specification for {} of {}'.\
                                          format(port_connect_spec, projection_socket, mech.name)

                        # Assign port along with dict's default values to tuple
                        port_connect_spec = (port,
                                              connection[WEIGHT],
                                              connection[EXPONENT],
                                              connection[PROJECTION])

                    # Dict specification for port itself
                    elif isinstance(port_connect_spec, dict):
                        # Get PORT entry
                        port_spec = port_connect_spec[PORT]
                        # Parse it to get reference to actual Port make sure it belongs to mech:
                        port = _get_port_for_socket(owner=owner,
                                                    port_spec=port_spec,
                                                    port_types=connect_with_attr,
                                                    mech=mech,
                                                    projection_socket=projection_socket)
                        if isinstance(port, list):
                            assert False, 'Got list of allowable ports for {} as specification for {} of {}'.\
                                           format(port_connect_spec, projection_socket, mech.name)
                        # Re-assign to PORT entry of dict (to preserve any other connection specifications in dict)
                        port_connect_spec[PORT] = port

                    # Tuple specification for Port itself
                    elif isinstance(port_connect_spec, tuple):
                        # Get PORT entry
                        port_spec = port_connect_spec[0]
                        # Parse it to get reference to actual Port make sure it belongs to mech:
                        port = _get_port_for_socket(owner=owner,
                                                    port_spec=port_spec,
                                                    port_types=connect_with_attr,
                                                    mech=mech,
                                                    projection_socket=projection_socket)
                        if isinstance(port, list):
                            assert False, 'Got list of allowable ports for {} as specification for {} of {}'.\
                                           format(port_connect_spec, projection_socket, mech.name)
                        # Replace parsed value in original tuple, but...
                        #    tuples are immutable, so have to create new one, with port_spec as (new) first item
                        # Get items from original tuple
                        port_connect_spec_tuple_items = [item for item in port_connect_spec]
                        # Replace port_spec
                        port_connect_spec_tuple_items[0] = port
                        # Reassign to new tuple
                        port_connect_spec = tuple(port_connect_spec_tuple_items)

                    # Recusively call _parse_connection_specs to get ProjectionTuple and append to connect_with_ports
                    connect_with_ports.extend(_parse_connection_specs(connectee_port_type, owner, port_connect_spec))

        # Process tuple, including final validation of Port specification
        # Tuple could be:
        #     (port_spec, projection_spec) or
        #     (port_spec, weight, exponent, projection_spec)
        # Note:  this is NOT the same as the Port specification tuple (which can have a similar format);
        #        the weights and exponents here specify *individual* Projections to a particular port,
        #            (vs. weights and exponents for an entire port, such as for InputPort);
        #        Port specification tuple is handled in the _parse_port_specific_specs() method of Port subclasses

        elif isinstance(connection, tuple):

            # 2-item tuple: can be (<value>, <projection_spec>) or (<port name or list of port names>, <Mechanism>)
            mech=None

            if len(connection) == 2:
                first_item, last_item = connection
                weight = DEFAULT_WEIGHT
                exponent = DEFAULT_EXPONENT
            elif len(connection) == 4:
                first_item, weight, exponent, last_item = connection
            else:
                raise ProjectionError("{} specification tuple for {} ({}) must have either two or four items".
                                      format(connectee_port_type.__name__, owner.name, connection))

            # Default assignments, possibly overridden below
            port_spec = first_item
            projection_spec = last_item

            # (<value>, <projection_spec>)
            if is_numeric(first_item):
                projection_spec = first_item

            # elif is_matrix(first_item):
            #     projection_spec = last_item
            #     port_spec = None

            elif _is_projection_spec(last_item):

                # If specification is a list of Ports and/or Mechanisms, get Projection spec for each
                if isinstance(first_item, list):
                     # Call _parse_connection_spec for each Port or Mechanism, to generate a conection spec for each
                    for connect_with_spec in first_item:
                        if not isinstance(connect_with_spec, (Port, Mechanism)):
                            raise PortError(f"Item in the list used to specify a {last_item.__name__} "
                                            f"for {owner.name} ({connect_with_spec.__name__}) "
                                            f"is not a {Port.__name__} or {Mechanism.__name__}")
                        c = _parse_connection_specs(connectee_port_type=connectee_port_type,
                                                    owner=owner,
                                                    connections=ProjectionTuple(connect_with_spec,
                                                                                weight, exponent,
                                                                                last_item))
                        connect_with_ports.extend(c)
                    # Move on to other connections
                    continue
                # Otherwise, go on to process this Projection specification
                port_spec = first_item
                projection_spec = last_item


            # (<port name or list of port names>, <Mechanism>)
            elif isinstance(first_item, (str, list)):
                port_item = first_item
                mech_item = last_item

                if not isinstance(mech_item, Mechanism):
                    raise ProjectionError("Expected 2nd item of the {} specification tuple for {} ({}) to be a "
                                          "Mechanism".format(connectee_port_type.__name__, owner.name, mech_item))
                # First item of tuple is a list of Port names, so recursively process it
                if isinstance(port_item, list):
                     # Call _parse_connection_spec for each Port name, to generate a conection spec for each
                    for port_Name in port_item:
                        if not isinstance(port_Name, str):
                            raise ProjectionError("Expected 1st item of the {} specification tuple for {} ({}) to be "
                                                  "the name of a {} of its 2nd item ({})".
                                                  format(connectee_port_type.__name__, owner.name, port_Name,
                                                         connects_with, mech_item.name))
                        c = _parse_connection_specs(connectee_port_type=connectee_port_type,
                                                    owner=owner,
                                                    connections=ProjectionTuple(port_Name,
                                                                                weight, exponent,
                                                                                mech_item))
                        connect_with_ports.extend(c)
                    # Move on to other connections
                    continue
                # Otherwise, go on to process (<Port name>, Mechanism) spec
                port_spec = port_item
                projection_spec = None
                mech=mech_item

            # Validate port specification, and get actual port referenced if it has been instantiated
            try:
                # FIX: 11/28/17 HACK TO DEAL WITH GatingSignal Projection to OutputPort
                # FIX: 5/11/19: CORRECTED TO HANDLE ControlMechanism SPECIFIED FOR GATING
                if ((_is_gating_spec(first_item) or _is_control_spec(first_item))
                    and (isinstance(last_item, OutputPort) or last_item == OutputPort)
                ):
                    projection_socket = SENDER
                    port_types = [OutputPort]
                    mech_port_attribute = [OUTPUT_PORTS]
                else:
                    port_types = connects_with
                    mech_port_attribute=connect_with_attr

                port = _get_port_for_socket(owner=owner,
                                              connectee_port_type=connectee_port_type,
                                              port_spec=port_spec,
                                              port_types=port_types,
                                              mech=mech,
                                              mech_port_attribute=mech_port_attribute,
                                              projection_socket=projection_socket)
            except PortError as e:
                raise ProjectionError(f"Problem with specification for {Port.__name__} in {Projection.__name__} "
                                      f"specification{(' for ' + owner.name) if owner else ' '}: " + e.error_value)

            # Check compatibility with any Port(s) returned by _get_port_for_socket

            if isinstance(port, list):
                ports = port
            else:
                ports = [port]

            for item in ports:
                if inspect.isclass(item):
                    port_type = item
                else:
                    port_type = item.__class__

                # # Test that port_type is in the list for port's connects_with
                from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal

                # KAM 7/26/18 modified to allow ControlMechanisms to be terminal nodes of compositions
                # We could only include ControlSignal in the allowed types if the receiver is a CIM?
                allowed = connects_with + modulators + [ControlSignal]

                if not any(issubclass(connects_with_port, port_type)
                           for connects_with_port in allowed):
                    spec = projection_spec or port_type.__name__
                    raise ProjectionError(f"Projection specification (\'{spec}\') for an incompatible connection: "
                                          f"{port_type.__name__} with {connectee_port_type.__name__} of {owner.name};"
                                          f" spec should be one of the following: "
                                          f"{' or '.join([r for r in port_type.canReceive])}, "
                                          f" or connectee should be one of the following: "
                                          f"{' or '.join([c.__name__ for c in connects_with])},")

            # Parse projection specification into Projection specification dictionary
            # Validate projection specification
            if _is_projection_spec(projection_spec) or _is_modulatory_spec(projection_spec) or projection_spec is None:

                # FIX: 11/21/17 THIS IS A HACK TO DEAL WITH GatingSignal Projection TO InputPort or OutputPort
                from psyneulink.core.components.ports.inputport import InputPort
                from psyneulink.core.components.ports.outputport import OutputPort
                from psyneulink.core.components.ports.modulatorysignals.gatingsignal import GatingSignal
                from psyneulink.core.components.projections.modulatory.gatingprojection import GatingProjection
                from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
                if (not isinstance(projection_spec, GatingProjection)
                    and isinstance(port, GatingSignal)
                    and connectee_port_type in {InputPort, OutputPort}):
                    projection_spec = port
                if (
                        (not isinstance(projection_spec, GatingProjection)
                         and port.__class__ == GatingSignal
                         and connectee_port_type in {InputPort, OutputPort})
                # # MODIFIED 9/27/19 NEW: [JDC]
                #     or
                #         (not isinstance(projection_spec, ControlProjection)
                #          and port.__class__ == ControlSignal
                #          and connectee_port_type in {InputPort, OutputPort})
                ):
                    projection_spec = port

                elif (_is_gating_spec(first_item) or _is_control_spec((first_item))
                      and not isinstance(last_item, (GatingProjection, ControlProjection))):
                    projection_spec = first_item
                projection_spec = _parse_projection_spec(projection_spec,
                                                         owner=owner,
                                                         port_type=connectee_port_type)

                _validate_connection_request(owner,
                                             connects_with + modulators,
                                             projection_spec,
                                             projection_socket,
                                             connectee_port_type)
            else:
                raise ProjectionError("Invalid {} specification ({}) for connection "
                                      "between {} \'{}\' and {} of \'{}\'.".
                                 format(Projection.__name__,
                                        projection_spec,
                                        port_type.__name__,
                                        port.name,
                                        connectee_port_type.__name__,
                                        owner.name))

            connect_with_ports.extend([ProjectionTuple(port, weight, exponent, projection_spec)])

        else:
            raise ProjectionError("Unrecognized, invalid or insufficient specification of connection for {}: \'{}\'".
                                  format(owner.name, connection))

    if not all(isinstance(projection_tuple, ProjectionTuple) for projection_tuple in connect_with_ports):
        raise ProjectionError("PROGRAM ERROR: Not all items are ProjectionTuples for {}".format(owner.name))

    return connect_with_ports

@tc.typecheck
def _validate_connection_request(
        owner,                                   # Owner of Port seeking connection
        connect_with_ports:list,                # Port to which connection is being sought
        projection_spec:_is_projection_spec,     # projection specification
        projection_socket:str,                   # socket of Projection to be connected to target port
        connectee_port:tc.optional(type)=None): # Port for which connection is being sought
    """Validate that a Projection specification is compatible with the Port to which a connection is specified

    Carries out undirected validation (i.e., without knowing whether the connectee is the sender or receiver).
    Use _validate_receiver or ([TBI] validate_sender) for directed validation.
    Note: connectee_port is used only for name in errors

    If projection_spec is a Projection:
        - if it is instantiated, compare the projection_socket specified (sender or receiver) with connect_with_port
        - if it in deferred_init, check to see if the specified projection_socket has been specified in _init_args;
            otherwise, use Projection's type
    If projection_spec is a class specification, use Projection's type
    If projection_spec is a dict:
        - check if there is an entry for the socket and if so, use that
        - otherwise, check to see if there is an entry for the Projection's type

    Returns:
        `True` if validation has been achieved to same level (though possibly with warnings);
        `False` if validation could not be done;
        raises an exception if an incompatibility is detected.
    """


    if connectee_port:
        connectee_str = " {} of".format(connectee_port.__name__)
    else:
        connectee_str = ""

    # Convert connect_with_ports (a set of classes) into a tuple for use as second arg in isinstance()
    connect_with_ports = tuple(connect_with_ports)
    # Make sure none of its entries are None (which will fail in isinstance()):
    if None in connect_with_ports:
        raise ProjectionError("PROGRAM ERROR: connect_with_ports ({}) should not have any entries that are \'None\'; "
                              "Check assignments to \'connectsWith' and \'modulators\' for each Port class".
                              format(connect_with_ports))

    connect_with_port_Names = ", ".join([c.__name__ for c in connect_with_ports if c is not None])

    # Used below
    def _validate_projection_type(projection_class):
        # Validate that Projection's type can connect with a class in connect_with_ports
        if any(port.__name__ in getattr(projection_class.sockets, projection_socket) for port in connect_with_ports):
            return True
        else:
            return False

    # If it is an actual Projection
    if isinstance(projection_spec, Projection):

        # It is in deferred_init status
        if projection_spec.initialization_status == ContextFlags.DEFERRED_INIT:

            # Try to get the Port to which the Projection will be connected when fully initialized
            #     as confirmation that it is the correct type for port_type
            try:
                projection_socket_port = projection_spec.socket_assignments[projection_socket]
            # Port for projection's socket couldn't be determined
            except KeyError:
                # Use Projection's type for validation
                # At least validate that Projection's type can connect with a class in connect_with_ports
                return _validate_projection_type(projection_spec.__class__)
                    # Projection's socket has been assigned to a Port
            else:
                # if both SENDER and RECEIVER are specified:
                if projection_spec._init_args[SENDER] and projection_spec._init_args[RECEIVER]:
                    # Validate that the Port is a class in connect_with_ports
                    if (isinstance(projection_socket_port, connect_with_ports) or
                            (inspect.isclass(projection_socket_port)
                             and issubclass(projection_socket_port, connect_with_ports))):
                        return True
                # Otherwise, revert again to validating Projection's type
                else:
                    return _validate_projection_type(projection_spec.__class__)

        # Projection has been instantiated
        else:
            # Determine whether the Port to which the Projection's socket has been assigned is in connect_with_ports
            # FIX: 11/4/17 - THIS IS A HACK TO DEAL WITH THE CASE IN WHICH THE connectee_port IS AN OutputPort
            # FIX:               THE projection_socket FOR WHICH IS USUALLY A RECEIVER;
            # FIX:           HOWEVER, IF THE projection_spec IS A GatingSignal
            # FIX:               THEN THE projection_socket MUST BE SENDER
            from psyneulink.core.components.ports.outputport import OutputPort
            from psyneulink.core.components.projections.modulatory.gatingprojection import GatingProjection
            from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
            if connectee_port is OutputPort and isinstance(projection_spec, (GatingProjection, ControlProjection)):
                projection_socket = SENDER
            projection_socket_port = getattr(projection_spec, projection_socket)
            if issubclass(projection_socket_port.__class__, connect_with_ports):
                return True

        # None of the above worked, so must be incompatible
        raise ProjectionError("{} specified to be connected with{} {} "
                              "is not consistent with the {} of the specified {} ({})".
                              format(Port.__name__, connectee_str, owner.name,
                                     projection_socket, Projection.__name__, projection_spec))

    # Projection class
    elif inspect.isclass(projection_spec) and issubclass(projection_spec, Port):
        if issubclass(projection_spec, connect_with_ports):
            return True
        raise ProjectionError("{} type specified to be connected with{} {} ({}) "
                              "is not compatible with the {} of the specified {} ({})".
                              format(Port.__name__, connectee_str, owner.name, projection_spec.__name__,
                                     projection_socket, Projection.__name__, connect_with_port_Names))

    # Port
    elif isinstance(projection_spec, Port):
        if isinstance(projection_spec, connect_with_ports):
            return True
        raise ProjectionError("{} specified to be connected with{} {} ({}) "
                              "is not compatible with the {} of the specified {} ({})".
                              format(Port.__name__, connectee_str, owner.name, projection_spec,
                                     projection_socket, Projection.__name__, connect_with_port_Names))

    # Port class
    elif inspect.isclass(projection_spec) and issubclass(projection_spec, Projection):
        _validate_projection_type(projection_spec)
        return True

    # Projection specification dictionary
    elif isinstance(projection_spec, dict):
        # Try to validate using entry for projection_socket
        if projection_socket in projection_spec and projection_spec[projection_socket] is not None:
            # Specification for the [projection_socket] entry (i.e., SENDER or RECEIVER)
            #    should be either of the correct class or a Mechanism
            #    (which assumes it will get properly resolved in context when the Projection is instantiated)
            if (projection_spec[projection_socket] in connect_with_ports or
                    isinstance(projection_spec[projection_socket], Mechanism)):
                return True
            else:
                raise ProjectionError("{} ({}) specified to be connected with{} {} is not compatible "
                                      "with the {} ({}) in the specification dict for the {}.".
                                      format(Port.__name__,
                                             connect_with_port_Names,
                                             connectee_str,
                                             owner.name,
                                             projection_socket,
                                             projection_spec[projection_socket],
                                             Projection.__name__))
        # Try to validate using entry for Projection' type
        elif PROJECTION_TYPE in projection_spec and projection_spec[PROJECTION_TYPE] is not None:
            _validate_projection_type(projection_spec[PROJECTION_TYPE])
            return True

    # Projection spec is too abstract to validate here
    #    (e.g., value or a name that will be used in context to instantiate it)
    return False

def _get_projection_value_shape(sender, matrix):
    """Return shape of a Projection's value given its sender and matrix"""
    from psyneulink.core.components.functions.transferfunctions import get_matrix
    matrix = get_matrix(matrix)
    return np.zeros(matrix.shape[np.atleast_1d(sender.value).ndim :])

# IMPLEMENTATION NOTE: MOVE THIS TO ModulatorySignals WHEN THAT IS IMPLEMENTED
@tc.typecheck
def _validate_receiver(sender_mech:Mechanism,
                       projection:Projection,
                       expected_owner_type:type,
                       spec_type=None,
                       context=None):
    """Check that Projection is to expected_receiver_type.

    expected_owner_type must be a Mechanism or a Projection
    spec_type should be LEARNING_SIGNAL, CONTROL_SIGNAL or GATING_SIGNAL

    Note:  this is a "directed" validation;
           for undirected validation of a Projection, use _validate_projection_specification

    """
    spec_type = " in the {} arg ".format(spec_type) or ""

    if projection.initialization_status == ContextFlags.DEFERRED_INIT:
        # receiver = projection._init_args['receiver'].owner
        port = projection._init_args['receiver']
        receiver = port.owner
    else:
        # receiver = projection.receiver.owner
        port = projection.receiver
        receiver = port.owner

    if isinstance(receiver, Mechanism):
        receiver_mech = receiver
    elif isinstance(receiver, Projection):
        receiver_mech = receiver.receiver.owner
    else:
        raise ProjectionError("receiver of projection ({}) must be a {} or {}".
                              format(projection.name, MECHANISM, PROJECTION))

    if not isinstance(receiver, expected_owner_type):
        raise ProjectionError("A {} specified {}for {} ({}) projects to a component other than the {} of a {}".
                                    format(projection.__class__.__name__,
                                           spec_type,
                                           sender_mech.name,
                                           receiver,
                                           port.__class__.__name__,
                                           expected_owner_type.__name__))

# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _add_projection_to(receiver, port, projection_spec, context=None):
    """Assign an "incoming" Projection to a receiver InputPort or ParameterPort of a Component object

    Verify that projection has not already been assigned to receiver;
        if it has, issue a warning and ignore the assignment request.

    Requirements:
       * receiver must be an appropriate Component object (currently, a Mechanism or a Projection);
       * port must be a specification of an InputPort or ParameterPort;
       * specification of InputPort can be any of the following:
                - INPUT_PORT - assigns projection_spec to (primary) InputPort;
                - InputPort object;
                - index for Mechanism.input_ports;
                - name of an existing InputPort (i.e., key for Mechanism.input_ports);
                - the keyword ADD_INPUT_PORT or the name for an InputPort to be added;
       * specification of ParameterPort must be a ParameterPort object
       * projection_spec can be any valid specification of a projection_spec
           (see `Port._instantiate_projections_to_port`).

    Args:
        receiver (Mechanism or Projection)
        port (Port subclass)
        projection_spec: (Projection, dict, or str)
        context

    """
    # IMPLEMENTATION NOTE:  ADD FULL SET OF ParameterPort SPECIFICATIONS
    #                       CURRENTLY, ASSUMES projection_spec IS AN ALREADY INSTANTIATED PROJECTION

    from psyneulink.core.components.ports.port import _instantiate_port
    from psyneulink.core.components.ports.port import Port_Base
    from psyneulink.core.components.ports.inputport import InputPort

    if not isinstance(port, (int, str, Port)):
        raise ProjectionError("Port specification(s) for {} (as receiver(s) of {}) contain(s) one or more items"
                             " that is not a name, reference to a {} or an index for one".
                              format(receiver.name, projection_spec.name, Port.__name__))

    # port is Port object, so use thatParameterPort
    if isinstance(port, Port_Base):
        return port._instantiate_projections_to_port(projections=projection_spec, context=context)

    # Generic INPUT_PORT is specified, so use (primary) InputPort
    elif port == INPUT_PORT:
        return receiver.input_port._instantiate_projections_to_port(projections=projection_spec, context=context)

    # input_port is index into input_ports OrderedDict, so get corresponding key and assign to input_port
    elif isinstance(port, int):
        try:
            key = receiver.input_ports[port]
        except IndexError:
            raise ProjectionError("Attempt to assign projection_spec ({0}) to InputPort {1} of {2} "
                                 "but it has only {3} input_ports".
                                 format(projection_spec.name, port, receiver.name, len(receiver.input_ports)))
        else:
            input_port = key

    # input_port is string (possibly key retrieved above)
    #    so try as key in input_ports OrderedDict (i.e., as name of an InputPort)
    if isinstance(port, str):
        try:
            return receiver.input_port[port]._instantiate_projections_to_port(projections=projection_spec, context=context)
        except KeyError:
            pass
        else:
            if receiver.prefs.verbosePref:
                warnings.warn("Projection_spec {0} added to {1} of {2}".
                              format(projection_spec.name, port, receiver.name))
            # return

    # input_port is either the name for a new InputPort or ADD_INPUT_PORT
    if not port == ADD_INPUT_PORT:
        if receiver.prefs.verbosePref:
            reassign = input("\nAdd new InputPort named {0} to {1} (as receiver for {2})? (y/n):".
                             format(input_port, receiver.name, projection_spec.name))
            while reassign != 'y' and reassign != 'n':
                reassign = input("\nAdd {0} to {1}? (y/n):".format(input_port, receiver.name))
            if reassign == 'n':
                raise ProjectionError("Unable to assign projection {0} to receiver {1}".
                                      format(projection_spec.name, receiver.name))

    # validate that projection has not already been assigned to receiver
    if receiver.verbosePref or projection_spec.sender.owner.verbosePref:
        if projection_spec in receiver.all_afferents:
            warnings.warn("Request to assign {} as projection to {} was ignored; it was already assigned".
                          format(projection_spec.name, receiver.owner.name))

    input_port = _instantiate_port(owner=receiver,
                                     port_type=InputPort,
                                     name=input_port,
                                     reference_value=projection_spec.value,
                                     reference_value_name='Projection_spec value for new InputPort',
                                     context=context)

    #  Update InputPort and input_ports
    if receiver.input_ports:
        receiver.parameters.input_ports._get(context)[input_port.name] = input_port

    # No InputPort(s) yet, so create them
    else:
        receiver.parameters.input_ports._set(
            ContentAddressableList(
                component_type=Port_Base,
                list=[input_port],
                name=receiver.name + '.input_ports'
            ),
            context
        )

    return input_port._instantiate_projections_to_port(projections=projection_spec, context=context)


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _add_projection_from(sender, port, projection_spec, receiver, context=None):
    """Assign an "outgoing" Projection from an OutputPort of a sender Mechanism

    projection_spec can be any valid specification of a projection_spec (see Port._instantiate_projections_to_port)
    port must be a specification of an OutputPort
    Specification of OutputPort can be any of the following:
            - OUTPUT_PORT - assigns projection_spec to (primary) OutputPort
            - OutputPort object
            - index for Mechanism OutputPorts OrderedDict
            - name of OutputPort (i.e., key for Mechanism.OutputPorts OrderedDict))
            - the keyword ADD_OUTPUT_PORT or the name for an OutputPort to be added

    Args:
        sender (Mechanism):
        projection_spec: (Projection, dict, or str)
        port (OutputPort, str, or value):
        context:
    """


    from psyneulink.core.components.ports.port import _instantiate_port
    from psyneulink.core.components.ports.port import Port_Base
    from psyneulink.core.components.ports.outputport import OutputPort

    # Validate that projection is not already assigned to sender; if so, warn and ignore

    if isinstance(projection_spec, Projection):
        projection = projection_spec
        if ((isinstance(sender, OutputPort) and projection.sender is sender) or
                (isinstance(sender, Mechanism) and projection.sender is sender.output_port)):
            if sender.verbosePref:
                warnings.warn("Request to assign {} as sender of {}, but it has already been assigned".
                              format(sender.name, projection.name))
                return

    if not isinstance(port, (int, str, OutputPort)):
        raise ProjectionError("Port specification for {0} (as sender of {1}) must be the name, reference to "
                              "or index of an OutputPort of {0} )".format(sender.name, projection_spec))

    # port is Port object, so use that
    if isinstance(port, Port_Base):
        port._instantiate_projection_from_port(projection_spec=projection_spec, receiver=receiver, context=context)
        return

    # Generic OUTPUT_PORT is specified, so use (primary) OutputPort
    elif port == OUTPUT_PORT:
        sender.output_port._instantiate_projections_to_port(projections=projection_spec, context=context)
        return

    # input_port is index into OutputPorts OrderedDict, so get corresponding key and assign to output_port
    elif isinstance(port, int):
        try:
            key = list(sender.output_ports.keys)[port]
        except IndexError:
            raise ProjectionError("Attempt to assign projection_spec ({0}) to OutputPort {1} of {2} "
                                 "but it has only {3} OutputPorts".
                                 format(projection_spec.name, port, sender.name, len(sender.output_ports)))
        else:
            output_port = key

    # output_port is string (possibly key retrieved above)
    #    so try as key in output_ports ContentAddressableList (i.e., as name of an OutputPort)
    if isinstance(port, str):
        try:
            sender.output_port[port]._instantiate_projections_to_port(projections=projection_spec, context=context)
        except KeyError:
            pass
        else:
            if sender.prefs.verbosePref:
                warnings.warn("Projection_spec {0} added to {1} of {2}".
                              format(projection_spec.name, port, sender.name))
            # return

    # output_port is either the name for a new OutputPort or ADD_OUTPUT_PORT
    if not port == ADD_OUTPUT_PORT:
        if sender.prefs.verbosePref:
            reassign = input("\nAdd new OutputPort named {0} to {1} (as sender for {2})? (y/n):".
                             format(output_port, sender.name, projection_spec.name))
            while reassign != 'y' and reassign != 'n':
                reassign = input("\nAdd {0} to {1}? (y/n):".format(output_port, sender.name))
            if reassign == 'n':
                raise ProjectionError("Unable to assign projection {0} to sender {1}".
                                      format(projection_spec.name, sender.name))

    output_port = _instantiate_port(owner=sender,
                                      port_type=OutputPort,
                                      name=output_port,
                                      reference_value=projection_spec.value,
                                      reference_value_name='Projection_spec value for new InputPort',
                                      context=context)
    #  Update output_port and output_ports
    try:
        sender.output_ports[output_port.name] = output_port
    # No OutputPort(s) yet, so create them
    except AttributeError:
        from psyneulink.core.components.ports.port import Port_Base
        sender.parameters.output_ports._set(
            ContentAddressableList(
                component_type=Port_Base,
                list=[output_port],
                name=sender.name + '.output_ports'
            )
        )

    output_port._instantiate_projections_to_port(projections=projection_spec, context=context)
