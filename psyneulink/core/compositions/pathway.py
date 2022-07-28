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

Pathway are specified as a list, each element of which is either a `Node <Composition_Nodes>` or set of Nodes,
possibly intercolated with specifications of `Projections <Projection>` between them.  `Nodes <Composition_Nodes>`
can be either a `Mechanism`, a `Composition`, or a tuple (Mechanism or Composition, `NodeRoles <NodeRole>`) that can
be used to assign `required_roles` to the Nodes in the Composition (see `Composition_Nodes` for additional details).
The Node(s) specified in each entry of the list project to the Node(s) specified in the next entry.

    .. _Pathway_Projection_List_Note:

    .. note::
       Only a *set* can be used to specify multiple Nodes for a given entry in a Pathway; a *list* can *not* be used
       for this purpose, as a list containing Nodes is always interpreted as a Pathway. If a list *is* included in a
       Pathway specification, then it and all other entries are considered as separate, parallel Pathways (see
       example *vii* in the `figure <Pathway_Figure>` below).

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

  .. _Pathway_Figure:

  .. figure:: _static/Pathways_fig.svg
     :scale: 50%

     **Examples of Pathway specifications** (including in the **pathways** argument of a `Composition`. *i)* Set
     of `Nodes <Composition_Nodes>`: each is treated as a `SINGLETON <NodeRole.SINGLETON>` within a single Pathway.
     *ii)* List of Nodes: forms a sequential Pathway. *iii)* Single Node followed by a set:  one to many mapping.
     *iv)* Set followed by a single Node: many to one mapping. *v)* Set followed by a set: many to many mapping.
     *vi)* Set followed by a list: because there is a list in the specification (``[C,D]``) all other entries are
     also treated as parallel Pathways (see `note <Pathway_Projection_List_Note>` above), so ``A`` and ``B`` in the
     set are `SINGLETON <NodeRole.SINGLETON>`\\s. *vii)* Set of Projections intercolated between two sets of Nodes:
     since the set of Projections does not include any involving ``B`` or ``E`` nor a default Projection specification,
     they are treated as `SINGLETON <NodeRole.SINGLETON>`\\s (compare with *x*). *viii)* Set followed by a Node and
     then a set:  many to one to many mapping. *ix)* Node followed by one that is a `nested Composition
     <Composition_Nested>` then another Node: one to many to one mapping. *x)* Set followed by a list of Projections
     then another set: since the list of Projections contains a default Projection specification (``matrix``)
     Projections are created between all pairings of nodes in the sets that precede and follow the list (compare with
     *vii*); note that the Projections must be specified in a list because the matrix is a list (or array), which
     cannot be included in a set (see `note <Pathway_Projection_Matrix_Note>` above).

     .. technical_note::
        The full code for the examples above can be found in `test_pathways_examples`,
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
      a pair of (sets of) Nodes (see `add_linear_processing_pathway <Composition.add_linear_processing_pathway>`
      for additional details).  The list must begin and end with a (set of) Node(s).
    ..
    * **set**: {`Node <Composition_Nodes>`, `Node <Composition_Nodes>`...} --
      each item of the set must be a `Node <Composition_Nodes>` (i.e., Mechanism or Composition, or a
      (`Mechanism <Mechanism>`, `NodeRoles <NodeRole>`) tuple);  each Node is treated as a `SINGLETON
      <NodeRole.SINGLETON>`.  Sets can also be used in a list specification (see above; and see
      `add_linear_processing_pathway <Composition.add_linear_processing_pathway>` for additional details).
    ..
    * **2-item tuple**: (Pathway, `LearningFunction`) -- used to specify a `learning Pathway
      <Composition_Learning_Pathway>`;  the 1st item must be one of the forms of Pathway specification
      described above, and the 2nd item must be a subclass of `LearningFunction`.

.. _Pathway_Specification_Multiple:

*Multiple Pathway Specifications*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the forms of single Pathway specification `above <Pathway_Specification>`, where multiple Pathways
can be specified (e.g., the **pathways** argument of the constructor for a `Composition` or its `add_pathways
<Composition.add_pathways>` method), they can be specified in a list, in which each item of the list can be any of
the forms above, or one of the following:

    * **Pathway** object or constructor: Pathway(pathway=\ `Pathway specification <Pathway_Specification>`,...).
    ..
    .. _Pathway_Specification_Dictionary:
    * **dict**: {name : Pathway) -- in which **name** is a str and **Pathway** is a Pathway object or constuctor,
      or one of the standard `Pathway specifications <Pathway_Specification>` listed above.

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


.. _Pathway_Structure:

Structure
---------

.. _Pathway_Attribute:

A Pathway has the following primary attributes:

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
executed, by default in the order in which they appear in the `pathway <Pathway.pathway>` attribute;  however, this
can be modified by `Conditions <Condition>` added to the Composition's `scheduler <Composition.scheduler>`.

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
