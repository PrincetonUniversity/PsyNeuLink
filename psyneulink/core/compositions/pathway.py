# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************** Pathway **************************************************************

"""

Related
-------

* `NodeRoles <NodeRole>`
* `PathwayRoles <PathwayRole>`
* `Composition`

Contents
--------

  * `Pathway_Overview`
  * `Pathway_Creation`
      - `Pathway_Template`
      - `Pathway_Assignment_to_Composition`
      - `Pathway_Name`
      - `Pathway_Specification`
          - `Pathway_Specification_Formats`
          - `Pathway_Specification_Projections`
          - `Pathway_Specification_Multiple`
      - `Composition_Add_Nested`
  * `Pathway_Structure`
  * `Pathway_Execution`
  * `Pathway_Class_Reference`

.. _Pathway_Overview:

Overview
--------

A Pathway is a sequence of `Nodes <Composition_Nodes>` and `Projections <Projection>`. Generally, Pathways are assigned
to a `Compositions`, but a Pathway object can also be created on its and used as a template for specifying a Pathway for
a Composition, as described below (see `Pathways  <Composition_Pathways>` for additional information about Pathways in
Compositions).

.. _Pathway_Creation:

Creating a Pathway
------------------

Pathway objects are created in one of two ways, either using the constructor to create a `template <Pathway_Template>`,
or automatically when a Pathway is `assigned to a Composition <Pathway_Assignment_to_Composition>`.

.. _Pathway_Template:

*Pathway as a Template*
~~~~~~~~~~~~~~~~~~~~~~~

A Pathway created on its own, using its constructor, is a **template**, that can be used to `specify a Pathway
<Pathway_Specification>` for one or more Compositions, as described `below <Pathway_Assignment_to_Composition>`;
however, it cannot be executed on its own.  When a Pathway object is used to assign a Pathway to a Composition,
its `pathway <Pathway.pathway>` attribute, and its `name <Pathway.name>` if that is not otherwise specified (see
`below <Pathway_Name>`), are used as the specification to create a new Pathway that is assigned to the Composition,
and the template remains unassigned.

.. _Pathway_Assignment_to_Composition:

*Assigning Pathways to a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pathways can be assigned to a `Composition` using the **pathways** argument of the Composition's constructor, or one of
its `Pathway addition methods <Composition_Pathway_Addition_Methods>`.  A Pathway object can be used, along with any
other form of `Pathway specification <Pathway_Specification>` for those arguments.  However, the Pathway object itself
is not assigned to the Composition; rather, it serves as a `template  <Pathway_Template>` for a new Pathway object
that is automatically created and assigned to the Composition.  The Composition's `Pathway addition methods
<Composition_Pathway_Addition_Methods>` return a Pathway object for each Pathway created and assigned to theComposition.
All of the Pathways assigned to a Composition are listed in its `pathways <Composition.pathways>` attribute.

.. _Pathway_Name:

*Pathway Name*
~~~~~~~~~~~~~~

If the **name** argument of the Pathway's constructor is used to assign it a name, this is used as the name of the
Pathway created when it is assigned to a Composition in its constructor, or using its `add_pathways
<Composition.add_pathways>` method.  This is also the case if one of the Composition's other `Pathway addition methods
<Composition_Pathway_Addition_Methods>` is used, as long as the **name** argument of those methods is not specified.
However, if the **name** argument is specified in those methods, or `Pathway specification dictionary
<Pathway_Specification_Dictionary>` is used to specify the Pathway's name, that takes precedence over, and replaces
one specified in the Pathway `template's <Pathway_Template>` `name <Pathway.name>` attribute.


.. _Pathway_Specification:

*Pathway Specification*
~~~~~~~~~~~~~~~~~~~~~~~

A Pathway is specified as a list, each element of which is either a `Node <Composition_Nodes>` or set of Nodes,
possibly intercolated with specifications of `Projections <Projection>` between them.  `Nodes <Composition_Nodes>`
can be either a `Mechanism`, a `Composition`, or a tuple (Mechanism or Composition, `NodeRoles <NodeRole>`) that can
be used to assign `required_roles` to the Nodes in the Composition (see `Composition_Nodes` for additional details).
The Node(s) specified in each entry of the list project to the Node(s) specified in the next entry.  In general, this
also determines their order of execution (though see `Pathway_Execution` for exceptions).

    .. _Pathway_Projection_Nesting_Warning:

    .. warning::
       Pathways can *not* be nested; that is, the list used to specify an individual Pathway cannot itself
       contain any lists that contain Nodes -- doing generates an error. A *set* can be used to specify
       multiple Nodes for a given entry in a Pathway, as described under `Pathway_Specification_Formats`;
       a list (or a set) can be used to specify multiple *Projections* in a Pathway, as described under
       `Pathway_Specification_Projections`; and, in some places, a list can be used to specify multiple
       *Pathways*, as described under `Pathway_Specification_Multiple`. However, the specification of an
       individual Pathway can *never* include a list that contains any Nodes.

    .. technical_note::
       The prohibition of nested Pathways should not pose a problem since, with the exception of sets of
       Nodes, the items in a Pathway are always executed sequentially, so it should be possible to achieve
       the same effect of a sequence of nested lists by including the elements within them in the same order
       all within a single outer list. Specifying pathways for parallel execution can be achieved by specifying
       multiple pathways in the **pathway** argument of a Composition's constructor, or its `add_pathways
       <Composition.add_pathways>` method, as described under `Pathway_Specification_Multiple` below.

# FIX: 4/13/22 - FINISH:
CF RELEVANT DISCUSSIONS IN Composition DOCSTRING
*Branching*.
      - local branching allowed, but not open-ended (sub-branching) -- violates idea of a defined "pathway":
        - allowed A -> {B,C} -> D -> {E,F}
        - not allowed: A -> B -> {C, D} -> C -> E, D -> E)

*Recurrence*.
      - can be recurrent, both locally within (A -> B <-> C -> D) and wider reaching A -> B-> C->)

*Overlap*.
      - note re: using same NODES = overlapping w/ example (Stroop) -- see Multiple Pathways below


All of these configurations are achieved and/or can be customized by the assignment of Projections within a Pathways,
as described below.

.. _Pathway_Specification_Projections:

*Pathway Projection Specifications*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Where no Projections are specified between entries in the list, default Projections (using a `FULL_CONNECTIVITY_MATRIX`;
see `MappingProjection_Matrix_Specification`) are created from each Node in the first entry, as the sender(s),
to each Node in the second, as receiver(s) (described further `below <Pathway_Projections>`).  Projections between
Nodes in the two entries can also be specified explicitly, by intercolating a Projection or set of Projections between
the two entries in the list.  If the sender and receiver are both a single Mechanism, then a single `MappingProjection`
can be `specified<MappingProjection_Creation>` between them.  The same applies if the sender is a `Composition` with
a single `OUTPUT <NodeRole.OUTPUT>` Node and/or the receiver is a `Composition` with a single `INPUT <NodeRole.INPUT>`
Node.  If either is a set of Nodes, or is a `nested Composition <Composition_Nested>` with more than one `INPUT
<NodeRole.INPUT>` or `OUTPUT <NodeRole.OUTPUT>` Node, respectively, then a collection of Projections can be specified
between any or all pairs of the Nodes in the set(s) and/or nested Composition(s), using either a set or list of
Projections (order of specification does not matter whether a set or a list is used). The collection can contain
`MappingProjections <MappingProjection>` between a specific pairs of Nodes and/or a single default specification
(either a `matrix <MappingProjection.matrix>` specification or a MappingProjection without any `sender
<MappingProjection.sender>` or `receiver <MappingProjection.receiver>` specified).

    .. _Pathway_Projection_Matrix_Note:

    .. note::
       If a collection of Projection specifications includes a default matrix specification, then a list must be used
       to specify the collection and *not* a set (since a matrix is unhashable and thus cannot be included in a set).

If a default Projection specification is included in the set, it is used to implement a Projection between any pair
of Nodes for which no MappingProjection is otherwise specified, whether within the collection or on its own; if no
Projections are specified for any individual pairs, a default Projection is created for every pairing of senders and
receivers. If a collection contains Projections for one or more pairs of Nodes, but does not include a default
projection specification, then no Projection is created between any of the other pairings.

If a pair of entries in a pathway has multiple sender and/or receiver Nodes specified (either in a set and/or belonging
to `nested Composition <Composition_Nested>`, and either no Projection(s) or only a default Projection is intercollated
between them, then a default set of Projections is constructed (using the default Projection specification, if provided)
between each pair of sender and receiver Nodes in the set(s) or nested Composition(s), as follows:

.. _Pathway_Projections:

* *One to one* - if both the sender and receiver entries are Mechanisms, or if either is a Composition and the
  sender has a single `OUTPUT <NodeRole.OUTPUT>` Node and the receiver has a single `INPUT <NodeRole.INPUT>`
  Node, then a default `MappingProjection` is created from the `primary OutputPort <OutputPort_Primary>` of the
  sender (or of its sole `OUTPUT <NodeRole.OUTPUT>` Node, if the sender is a Composition) to the `primary InputPort
  <InputPort_Primary>` of the receiver (or of its sole of `INPUT <NodeRole.INPUT>` Node, if the receiver is
  a Composition), and the Projection specification is intercolated between the two entries in the `Pathway`.

* *One to many* - if the sender is either a Mechanism or a Composition with a single `OUTPUT <NodeRole.OUTPUT>` Node,
  but the receiver is either a Composition with more than one `INPUT <NodeRole.INPUT>` Node or a set of Nodes, then
  a `MappingProjection` is created from the `primary OutputPort <OutputPort_Primary>` of the sender Mechanism (or of
  its sole `OUTPUT <NodeRole.OUTPUT>` Node if the sender is a Composition) to the `primary InputPort
  <InputPort_Primary>` of each `INPUT <NodeRole.OUTPUT>` Node of the receiver Composition and/or Mechanism in the
  receiver set, and a set containing the Projections is intercolated between the two entries in the `Pathway`.

* *Many to one* - if the sender is a Composition with more than one `OUTPUT <NodeRole.OUTPUT>` Node or a set of
  Nodes, and the receiver is either a Mechanism or a Composition with a single `INPUT <NodeRole.INPUT>` `OUTPUT
  <NodeRole.OUTPUT>` Node in the Composition or Mechanism in the set of sender(s), to the `primary InputPort
  <InputPort_Primary>` of the receiver Mechanism (or of its sole `INPUT <NodeRole.INPUT>` Node if the receiver is
  a Composition), and a set containing the Projections is intercolated between the two entries in the `Pathway`.

* *Many to many* - if both the sender and receiver entries contain multiple Nodes (i.e., are sets,  and/or the
  the sender is a Composition that has more than one `INPUT <NodeRole.INPUT>` Node and/or the receiver has more
  than one `OUTPUT <NodeRole.OutPUT>` Node), then a Projection is constructed for every pairing of Nodes in the
  sender and receiver entries, using the `primary OutputPort <OutputPort_Primary>` of each sender Node and the
  `primary InputPort <InputPort_Primary>` of each receiver node.

|
      COMMENT:
      # FIX: ADD EXAMPLES WITH HOP AND RECURRENCE
      COMMENT

      .. _Pathway_Figure_Single:

      .. figure:: _static/Pathways_Single_fig.svg
         :scale: 50%

         **Examples of single Pathway specifications** (with alternate specifications shown under some configurations).
         *i)* Set of `Nodes <Composition_Nodes>`: each is treated as a `SINGLETON <NodeRole.SINGLETON>`, that are
         all executed in parallel.
         *ii)* List of Nodes: forms a sequence of Nodes that are executed in order.
         *iii)* List of Nodes with intercollated Projections that form a "hop": ``B`` Projects both to ``C`` and ``D``,
         so that when the latter executes it receives input from both ``B`` and ``C``.
         *iv)* List of Nodes with intercollated Projections that form a recurrence: the feedbak projection (``db``)
         forms a loop involving ``B``, ``C`` and ``D``; note that since that Projection is intercollated between ``D``
         and ``E``, the Projection from ``D`` to ``E`` (``de``) must also be explicitly specified or not Projection
         will be created between them.
         *v)* Single Node followed by a set: forms a one to many mapping, in which ``A`` executes first, followed by
         simultaneous execution of ``B`` and ``C``.
         *vi)* Set followed by a single Node: forms a many to one mapping, in which ``A`` and ``B`` execute
         simultaneously, followed by ``C``.
         *vii)* Set followed by a set: forms a many to many mapping, in which ``A`` and ``B`` execute simultaneously,
         followed by ``C`` and ``D``.  *vi)* Set followed by a Node and then a set:  many to one to many mapping.
         *viii)* Set followed by a Node and then another set: forms a many to one to many mapping.
         *ix)* Node followed by one that is a `nested Composition <Composition_Nested>` then another Node: forms a
         one to many to one mapping, from the outer Composition to the Nodes of the nested Composition, and then back
         out to the outer Composition.
         *x)* Set of Projections intercolated between two sets of Nodes: since the set of Projections does not
         include any involving ``B`` or ``E``, nor a default Projection specification, they are treated as
         `SINGLETON <NodeRole.SINGLETON>`\\s
         COMMENT:
         (compare with *ZZZ* in the `figure below <Pathway_Figure_Multiple>`)
         COMMENT
         , though all are within the same Pathway.
         *xi)* Set of Projections intercolated between two
         sets of Nodes:  same as to *x*, except that a default matrix is specified (``matrix``) in the set of
         Projections; as a consequence, Projections are created for all of the other pairings of Nodes, forming
         and all-to-all mapping (note that, in this case, the Projections must be specified in a list (rather than
         a set, as in *x*), since the matrix itself is a list, which cannot be included in a set; see `note
         <Pathway_Projection_Matrix_Note>` above).
         COMMENT:
         PUT IN EXAMPLE: Note that the value of the `matrix <MappingProjection.matrix>` for the default Projections
         is assigned ``[3]``, whereas the value of the matrix assigned to the ``af`` and ``cd`` Projections is ``[1]``
         (the default value of the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection` is not
         specified -- see `example <ZZZ>` below).
         COMMENT

         |
         .. technical_note::
            The full code for the examples above can be found in `test_pathway_figure_examples`,
            although some have been graphically rearranged for illustrative purposes.

.. _Pathway_Specification_Formats:

*Pathway Specification Formats*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following formats can be used to specify a Pathway in the **pathway** argument of the constructor for
the Pathway, the **pathways** argument of the constructor for a `Composition`, or the corresponding argument
of any of a Composition's `Pathway addition methods <Composition_Pathway_Addition_Methods>`:

    * `Node <Composition_Nodes>`: -- assigns the Node as `SINGLETON <NodeRole.SINGLETON>` in a Pathway.
    ..
    .. _Pathway_Specification_List:

    * **list**: [`Node <Composition_Nodes>`, <`Projection(s) <Projection>`,> `Node <Composition_Nodes>`...] --
      each item of the list must be a `Node <Composition_Nodes>` (i.e., Mechanism or Composition, or a
      (`Mechanism <Mechanism>`, `NodeRoles <NodeRole>`) tuple) or set of Nodes, optionally with a `Projection
      specification <Projection_Specification>`, a (`Projection specification <Projection_Specification>`,
      `feedback specification <Composition_Feedback_Designation>`) tuple, or a set of either interposed between
      a pair of (sets of) Nodes (see `Pathway_Specification_Projections` above for additional details). The list
      must begin and end with a (set of) Node(s).  By default, the elements of a list execute sequentially in the
      order determined their position in the list; however this can be modified by the use of `Conditions <Condition>`
      (see `Pathway_Execution` for an example).

    .. _Pathway_Specification_Set:

    * **set**: {`Node <Composition_Nodes>`, `Node <Composition_Nodes>`...} --
      each item of the set must be a `Node <Composition_Nodes>` (i.e., Mechanism or Composition, or a (`Mechanism
      <Mechanism>`, `NodeRoles <NodeRole>`) tuple);  each Node is treated as a `SINGLETON <NodeRole.SINGLETON>` and,
      by default, all are executed simultaneously (i.e., within the same `TIME_STEP <TimeScale.TIME_STEP>` of
      execution), though this can be modified by the use of `Conditions`. Sets of Nodes can also be specified within
      a list in which case, by default, they all execute at the same point in the sequence of execution specified by
      the list (again, subject to the specification of any relevant `Conditions <Condition>`).  Sets of Nodes can be
      combined with sets of Projections in a list to create a variety of configurations within a Pathway, as described
      above under `Pathway_Specification_Projections` (see `figure <Pathway_Figure_Single>` for examples)
    ..
    * **2-item tuple**: (Pathway, `LearningFunction`) -- used to specify a `learning Pathway
      <Composition_Learning_Pathway>`;  the 1st item must be one of the forms of Pathway specification
      described above, and the 2nd item must be a subclass of `LearningFunction`.

.. _Pathway_Specification_Multiple:

*Multiple Pathway Specifications*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple Pathways can be sepecified in the **pathways** argument of the constructor for a `Composition` or its
`add_pathways <Composition.add_pathways>` method, by including them in a list, in which each can be any of the forms
of Pathway specification described above above, or one of the ones listed below. Each item in the list constitutes
a separate Pathway, that are executed in parallel with one another (subject to any `Conditions <Condition>` specified
for their `Nodes <Composition_Nodes>`; see `Pathway_Execution`).

    .. _Pathway_Specification_Multiple_List:

    .. note::
       For convenience, the **pathways** argument of a Composition and its `add_pathways <Composition.add_pathways>`
       method permit a single pathway to be specified in a list without embedding it in an outer list.  However,
       this can *not* include any lists of Nodes, which would constitute nested Pathways that are not permitted (see
       `note <Pathway_Projection_Nesting_Warning>` above).  Accordingly, if any list is specified that contains any
       Nodes, then that -- and *all* other items in the list -- are interpreted as their own distinct Pathway
       specifications (see example ** in the `figure <Pathway_Figure_Multiple>` below).

In addition to a list, set or 2-item tuple, the following can also be used to specify a Pathway where multiple
Pathways can be specified:

    * **Pathway** object or constructor: Pathway(pathway=\ `Pathway specification <Pathway_Specification>`,...).
    ..
    .. _Pathway_Specification_Dictionary:
    * **dict**: {name : Pathway) -- in which **name** is a str and **Pathway** is a Pathway object or constuctor,
      or one of the standard `Pathway specifications <Pathway_Specification>` listed above.

    COMMENT:
    .. note::
       If any of the following is used to specify the **pathways** argument:
         * a **standalone** `Node <Composition_Nodes>` (i.e., not in a list), \n
         * a **single Node** alone in a list, \n
         * a **set** of Nodes, \n
         * one or more Nodes with any other form of `Pathway specification <Pathway_Specification>` in the list \n
       then each such Node in the list is assigned as a `SINGLETON <NodeRole.SINGLETON>` Node in its own Pathway.
       However, if the list contains only Nodes, then it is treated as a single Pathway (i.e., the list form of
       `Pathway specification <Pathway_Specification>` described above.  Thus:
         **pathway**: NODE -> single pathway \n
         **pathway**: [NODE] -> single pathway \n
         **pathway**: [NODE, NODE...] -> single pathway \n
         **pathway**: [NODE, () or {} or `Pathway`...] -> individual Pathways for each specification.
    COMMENT

    .. _Pathway_Figure_Multiple:

    .. figure:: _static/Pathways_Multiple_fig.svg
       :scale: 50%

       **Examples of multiple Pathway specifications**. *i)* Set followed by a list: because there is a list
       in the specification (``[C,D]``) all other entries are also treated as parallel Pathways (see `note
       <Pathway_Projection_Nesting_Warning>` above), so ``A`` and ``B`` in the set are `SINGLETON
       <NodeRole.SINGLETON>`\\s. *ii)* Set of Projections intercolated between two sets of Nodes: since the
       set of Projections does not include any involving ``B`` or ``E`` nor a default Projection specification,
       they are treated as `SINGLETON <NodeRole.SINGLETON>`\\s (compare with *x*).

# FIX: 4/13/22 - FINISH:
*Sequence*
      - can break a Pathway into sequence of Pathways;  useful if the are to be used in different ways in diff Comps?

*Overlap*.
      - note re: using same NODES = overlapping w/ example (Stroop) -- see Multiple Pathways below

.. _Pathway_Structure:

Structure
---------

.. _Pathway_Attribute:

A Pathway object has the following primary attributes:

* `pathway <Pathway.pathway>` - if the Pathway was created on its own, this contains the specification provided in
  the **pathway** arg of its constructor; that is, depending upon how it was specified, it may or may not contain
  fully constructed `Components <Component>`.  This is passed to the **pathways** argument of a Composition's
  constructor or one of its `pathway addition methods <Composition_Pathway_Addition_Methods>` when the Pathway is used
  in the specification of any of these.  In contrast, when a Pathway is created by a Composition (and assigned to its
  `pathways <Composition.pathways>` attribute), then the actual `Mechanism(s) <Mechanism>` and/or `Composition(s)`
  that comprise `Nodes <Composition_Nodes>`, and the `Projection(s) <Projection>` between them, are listed in the
  Pathway's `pathway <Pathway.pathway>` attribute.

* `composition <Pathway.composition>` - contains the `Composition` that created the Pathway and to which it belongs,
  or None if it is a ``template <Pathway_Template>` (i.e., was constructed on its own).

* `roles <Pathway.roles>` and `Node <Composition_Nodes>` attributes - if the Pathway was created by a Composition,
  the `roles <Pathway.roles>` attribute `this lists the `PathwayRoles <PathwayRole>` assigned to it by the Composition
  that correspond to the `NodeRoles <NodeRole`> of its Components, and the `Nodes <Composition_Nodes>` with each of
  those `NodeRoles <NodeRole>` is assigned to a corresponding attribute on the Pathway.  If the Pathway does not belong
  to a Composition (i.e., it is a `template <Pathway_Template>`), then these attributes return None.

* `learning_function <Pathway.learning_function>` - the LearningFunction assigned to the Pathway if it is a
  `learning Pathway <Composition_Learning_Pathway>` that belongs to a Composition; otherwise it is None.

.. _Pathway_Execution:

Execution
---------

A Pathway cannot be executed on its own.  Its Components are executed when the Composition to which it belongs is
executed, sequentially in the order in which they appear in the `pathway <Pathway.pathway>` attribute, with Nodes
specified within a `set <Pathway_Specification_Set>` all executed within the same `TIME_STEP <TimeScale.TIME_STEP>`.
However, the order can be modified by `Conditions <Condition>` assigned to the Composition's `scheduler
<Composition.scheduler>` (see `example <Pathway_Example_Conditions>` below).

.. _Pathway_Examples

Examples
--------

.. _Pathway_Projection_Example:
   SHOW CODE AND MATRIX VALUE ASSIGNEMENTS FOR EXAMPLE iX IN FIGURE_SINGLE

.. _Pathway_Example_Overlap:
   - STROOP EXAMPLE

.. _Pathway_Example_Recurrence:
   - CONFLICT MONITORING EXAMPLE?

.. _Pathway_Example_Overlapping
   - RUMELHART NETWORK

.. _Pathway_Example_Branching:
   - WITHIN PATHWAY AND ILLEGAL OPEN ENDED.

.. _Pathway_Example_Conditions:

The following example illustrates the use of `Conditions <Condition>` to modify the order of execution of `Nodes
<Composition_Nodes>` in a Pathway::

    >>> A = pnl.ProcessingMechanism(name='A')
    >>> B = pnl.ProcessingMechanism(name='B')
    >>> C = pnl.ProcessingMechanism(name='C')
    >>>
    >>> comp = pnl.Composition(pathways=[[A, B], [C]])
    >>> comp.scheduler.add_condition_set({ A: pnl.EveryNCalls(B, 1),
    ...                                    B: pnl.EveryNCalls(C, 1)})
    >>> comp.run(inputs={A: 0, C: 0})
    [array([0.]), array([0.])]
    >>> print(comp.scheduler.execution_list)
    {'Composition-0': [{(ProcessingMechanism C)}, {(ProcessingMechanism B)}, {(ProcessingMechanism C), (ProcessingMechanism A)}]}

``B`` follows ``A`` in the Pathway ``[A, B]`` specified for ``comp``, so ordinarily it would execute only *after*
``A`` each time that executes.  However, the `Condition` specified for ``A`` (``pnl.EveryNCalls(B, 1)``) causes it to
execute only after ``B`` executes.  Since the Condition specified for ``B`` makes it dependent on ``C``, both ``A``
and ``B`` wait until ``C`` has executed, after which the Condition ``B`` is satisfied so it executes, satisfying the
Condition for ``A`` which then executes, thus reversing the order of execution for ``A`` and ``B``.

.. _Pathway_Class_Reference:


Class Reference
---------------

"""
import warnings
from enum import Enum

import typecheck as tc

from psyneulink.core.components.shellclasses import Mechanism
from psyneulink.core.compositions.composition import Composition, CompositionError, NodeRole
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ANY, CONTEXT, FEEDBACK, MAYBE, NODE, LEARNING_FUNCTION, OBJECTIVE_MECHANISM, PROJECTION, TARGET_MECHANISM
from psyneulink.core.globals.registry import register_category

__all__ = [
    'Pathway', 'PathwayRegistry', 'PathwayRole'
]


PathwayRegistry= {}


def _is_pathway_entry_spec(entry, desired_type:tc.enum(NODE, PROJECTION, ANY)):
    """Test whether pathway entry is specified type (NODE or PROJECTION)"""
    from psyneulink.core.components.projections.projection import _is_projection_spec
    node_types = (Mechanism, Composition)
    is_node = is_proj = is_set = False

    if desired_type in {NODE, ANY}:
        is_node = (isinstance(entry, node_types)
                   or (isinstance(entry, tuple)
                       and isinstance(entry[0], node_types)
                       and (isinstance(entry[1], NodeRole) or
                            (isinstance(entry[1], list) and all(isinstance(nr, NodeRole) for nr in entry[1])))))

    if desired_type in {PROJECTION, ANY}:
        is_proj = (_is_projection_spec(entry)
                   or (isinstance(entry, tuple)
                       and _is_projection_spec(entry[0])
                       and entry[1] in {True, FEEDBACK, False, MAYBE})
                   or (isinstance(entry, (set,list))
                   # or (isinstance(entry, set)
                       and all(_is_projection_spec(item) for item in entry)))

    if desired_type in {ANY}:
        is_set = (isinstance(entry, set) and all(_is_node_spec(item) for item in entry))

    if is_node or is_proj or is_set:
        return True
    else:
        return False


def _is_node_spec(value):
    return _is_pathway_entry_spec(value, NODE)


class PathwayRole(Enum):
    """Roles assigned to the `Pathway` of a `Composition`.

    Attributes
    ----------

    ORIGIN
        A `Pathway` that includes an `ORIGIN` `Node <Composition_Nodes>` of the `Composition`.

    INPUT
        A `Pathway` that includes an `INPUT` `Node <Composition_Nodes>` of the `Composition`.

    SINGELTON
        A `Pathway` with a single `Node <Composition_Nodes>` that is a `SINGLETON` of the `Composition`.

    INTERNAL
        A `Pathway` that does not include any `ORIGIN` or `TERMINAL` `Nodes <Composition_Nodes>` of the `Composition`.

    OUTPUT
        A `Pathway` that includes an `OUTPUT` `Node <Composition_Nodes>` of the `Composition`.

    TERMINAL
        A `Pathway` that includes a `TERMINAL` `Node <Composition_Nodes>` of the `Composition`.

    CYCLE
        A `Pathway` that constitutes a `CYCLE` in the `Composition`.

    COMMENT:
    CONTROL
        A `Pathway` that constitutes a `control pathway <Composition_Control_Pathways>` of the `Composition`.
    COMMENT

    LEARNING
        A `Pathway` that constitutes a `learning Pathway <Composition_Learning_Pathway>` of the `Composition`.

    """
    ORIGIN = 0
    INPUT = 1
    SINGLETON = 2
    INTERNAL = 3
    OUTPUT = 4
    TERMINAL = 5
    CYCLE = 6
    CONTROL = 7
    LEARNING = 8


class PathwayError(Exception):

    def __init__(self, error_value, **kwargs):
        self.error_value = error_value
        self.return_items = kwargs

    def __str__(self):
        return repr(self.error_value)


class Pathway(object):
    """
    Pathway(      \
        pathway,  \
        name=None \
        )

    A sequence of `Nodes <Composition_Nodes>` and `Projections <Projection>` in a `Composition`, or a template
    for one that can be assigned to one or more Compositions.

    Arguments
    ---------

    pathway : list[`Node <Composition_Nodes>`, <`Projection <Projection>`,> `Node <Composition_Nodes>`...]
        specifies list of `Nodes <Composition_Nodes>` and intercolated `Projections <Projection>` to be
        created for the Pathway.

    name : str : default see `name <Pathway.name>`
        specifies the name of the Pathway;  see `name <Pathway.name>` for additional information.

    Attributes
    ----------

    pathway : `Node <Composition_Nodes>`, list, tuple, or dict.
        if the Pathway is created on its own, this contains the specification provided to the **pathway** argument
        of its constructor, and can take any of the forms permitted for `Pathway specification <Pathway_Specification>`;
        if the Pathway is created by a Composition, this is a list of the `Nodes <Pathway_Nodes>` and intercolated
        `Projections <Projection>` in the Pathway (see `above <Pathway_Attribute>` for additional details).

    composition : `Composition` or None
        `Composition` to which the Pathway belongs;  if None, then Pathway is a `template <Pathway_Template>`.

    roles : list[`PathwayRole`] or None
        list of `PathwayRole(s) <PathwayRole>` assigned to the Pathway, based on the `NodeRole(s) <NodeRole>`
        assigned to its `Nodes <Composition>` in the `composition <Pathway.composition>` to which it belongs.
        Returns an empty list if belongs to a Composition but no `PathwayRoles <PathwayRole>` have been assigned,
        and None if the Pathway is a `tempalte <Pathway_Template>` (i.e., not assigned to a Composition).

    learning_function : `LearningFunction` or None
        `LearningFunction` used by `LearningMechanism(s) <LearningMechanism>` associated with Pathway if
        it is a `learning pathway <Composition_Learning_Pathway>`.

    input : `Mechanism <Mechanism>` or None
        `INPUT` node if Pathway contains one.

    output : `Mechanism <Mechanism>` or None
        `OUTPUT` node if Pathway contains one.

    target : `Mechanism <Mechanism>` or None
        `TARGET` node if if Pathway contains one; same as `learning_components
        <Pathway.learning_components>`\\[*TARGET_MECHANISM*].

    learning_objective : `Mechanism <Mechanism>` or None
        `OBJECTIVE_MECHANISM` if Pathway contains one; same as `learning_components
        <Pathway.learning_components>`\\[*COMPATOR_MECHANISM*].

    learning_components : dict or None
        dict containing the following entries if the Pathway is a `learning Pathway <Composition_Learning_Pathway>`
        (and is assigned `PathwayRole.LEARNING` in `roles <Pathway.roles>`):

          *TARGET_MECHANISM*: `ProcessingMechanism` (assigned to `target <Pathway.target>`)
          ..
          *OBJECTIVE_MECHANISM*: `ComparatorMechanism` (assigned to `learning_objective <Pathway.learning_objective>`)
          ..
          *LEARNING_MECHANISMS*: `LearningMechanism` or list[`LearningMechanism`]
          ..
          *LEARNED_PROJECTIONS*: `Projection <Projection>` or list[`Projections <Projection>`]

        These are generated automatically and added to the `Composition` when the Pathway is assigned to it.
        Returns an empty dict if it is not a `learning Pathway <Composition_Learning_Pathway>`, and None
        if the Pathway is a `tempalte <Pathway_Template>` (i.e., not assigned to a Composition).

    name : str
        the name of the Pathway; if it is not specified in the **name** argument of the constructor, a default is
        assigned by PathwayRegistry (see `Registry_Naming` for conventions used for default and duplicate names).
        See `note <Pathway_Name_Note>` for additional information.

    """
    componentType = 'Pathway'
    componentName = componentType
    name = componentName

    @handle_external_context()
    def __init__(
            self,
            pathway:list,
            name=None,
            **kwargs
    ):

        context = kwargs.pop(CONTEXT, None)

        # Get composition arg (if specified)
        self.composition = None
        self.composition = kwargs.pop('composition',None)
        # composition arg not allowed from command line
        if self.composition and context.source == ContextFlags.COMMAND_LINE:
            raise CompositionError(f"'composition' can not be specified as an arg in the constructor for a "
                              f"{self.__class__.__name__}; it is assigned when the {self.__class__.__name__} "
                              f"is added to a {Composition.__name__}.")
        # composition arg must be a Composition
        if self.composition:
            assert isinstance(self.composition, Composition), \
                f"'composition' arg of constructor for {self.__class__.__name__} must be a {Composition.__name__}."

        # There should be no other arguments in constructor
        if kwargs:
            raise CompositionError(f"Illegal argument(s) used in constructor for {self.__class__.__name__}: "
                                   f"{', '.join(list(kwargs.keys()))}.")

        # Register and get name
        # - if called from command line, being used as a template, so don't register
        if context.source == ContextFlags.COMMAND_LINE:
            # But do pass through name so that it can be used to construct the instance that will be used
            self.name = name
        else:
            # register and set name
            register_category(
                entry=self,
                base_class=Pathway,
                registry=PathwayRegistry,
                name=name
            )

        # Validate entries (no lists allowed for Nodes -- only allowed for Projections)
        bad_entries = [repr(entry) for entry in pathway
                       if isinstance(entry,list) and not _is_pathway_entry_spec(entry, PROJECTION)]
        if bad_entries:
            raise PathwayError(f"A Pathway cannot contain an embedded list with Nodes (use set): "
                               f"{', '.join(bad_entries)}")

        # Initialize attributes
        self.pathway = pathway
        if self.composition:
            self.learning_components = {}
            self.roles = set()
        else:
            self.learning_components = None
            self.roles = None

    def _assign_roles(self, composition):
        """Assign `PathwayRoles <PathwayRole>` to Pathway based `NodeRoles <NodeRole>` assigned to its `Nodes
        <Composition_Nodes>` by the **composition** to which it belongs.
        """
        assert self.composition, f'_assign_roles() cannot be called for {self.name} ' \
                                 f'because it has not been assigned to a {Composition.__name__}.'
        self.roles = set()
        for node in self.pathway:
            if not isinstance(node, (Mechanism, Composition)):
                continue
            roles = composition.get_roles_by_node(node)
            if NodeRole.ORIGIN in roles:
                self.roles.add(PathwayRole.ORIGIN)
            if NodeRole.INPUT in roles:
                self.roles.add(PathwayRole.INPUT)
            if NodeRole.SINGLETON in roles and len(self.pathway)==1:
                self.roles.add(PathwayRole.SINGLETON)
            if NodeRole.OUTPUT in roles:
                self.roles.add(PathwayRole.OUTPUT)
            if NodeRole.TERMINAL in roles:
                self.roles.add(PathwayRole.TERMINAL)
            if NodeRole.CYCLE in roles:
                self.roles.add(PathwayRole.CYCLE)
        if not [role in self.roles for role in {PathwayRole.ORIGIN, PathwayRole.TERMINAL}]:
            self.roles.add(PathwayRole.INTERNAL)
        if self.learning_components:
            self.roles.add(PathwayRole.LEARNING)

    @property
    def input(self):
        if self.composition and PathwayRole.INPUT in self.roles:
            input_node = next(n for n in self.pathway if n in self.composition.get_nodes_by_role(NodeRole.INPUT))
            if input_node:
                return input_node
            else:
                assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                              f"is assigned PathwayRole.INPUT but has no INPUT node."

    @property
    def output(self):
        if self.composition and PathwayRole.OUTPUT in self.roles:
            output_node = next(n for n in self.pathway if n in self.composition.get_nodes_by_role(NodeRole.OUTPUT))
            if output_node:
                return output_node
            else:
                assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                              f"is assigned PathwayRole.OUTPUT but has no OUTPUT node."

    @property
    def target(self):
        if self.composition:
            try:
                return self.learning_components[TARGET_MECHANISM]
            except:
                if PathwayRole.LEARNING not in self.roles:
                    warnings.warn(f"{self.__class__.__name__} {self.name} 'target' attribute is None "
                                  f"because it is not a learning_pathway.")
                else:
                    assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                                  f"has PathwayRole.LEARNING assigned but no 'target' attribute."
                return None

    @property
    def learning_objective(self):
        if self.composition:
            try:
                return self.learning_components[OBJECTIVE_MECHANISM]
            except:
                if PathwayRole.LEARNING not in self.roles:
                    warnings.warn(f"{self.__class__.__name__} {self.name} 'learning_objective' attribute "
                                  f"is None because it is not a learning_pathway.")
                else:
                    assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                                  f"has PathwayRole.LEARNING assigned but no 'learning_objective' attribute."
                return None

    @property
    def learning_function(self):
        if self.composition:
            try:
                return self.learning_components[LEARNING_FUNCTION]
            except:
                if PathwayRole.LEARNING not in self.roles:
                    warnings.warn(f"{self.__class__.__name__} {self.name} 'learning_function' attribute "
                                  f"is None because it is not a learning_pathway.")
                else:
                    assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                                  f"has PathwayRole.LEARNING assigned but no 'learning_function' attribute."
                return None
