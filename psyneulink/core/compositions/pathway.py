# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************** Pathway **************************************************************

"""

.. _Pathway_Overview:

Overview
--------

A Pathway is a sequence of `Nodes <Composition_Nodes>` and `Projections <Projection>` in a `Composition`. Pathways
are created and added to a `Composition` if the **pathways** argument of the Composition's constructor is specified,
and/or whenever the Composition's `pathway creation methods <Composition_Pathway_Methods>` are used.  A Pathway can
also be created on its own, and used in the **pathway** argument of the Composition's constructor or one of its
`pathway creation methods <Composition_Pathway_Methods>` methods.  Although Pathways are not required in Compositions,
they are useful for constructing them, and are required to implement `learning <Composition_Learning>` in a Composition.

Creating a Pathway
------------------

A Pathway can be created as part of a Composition or on its own.  If the **composition** argument of the
Pathway's constructor is specified, it is added to the specified `Composition`;  if that is not specified,
then a standalone Pathway is created that can be used as a template for specifiying the pathway of one or
more Compositions, as described below.

*Assigning to a Pathway to a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the **composition** argument is specified in the Pathway's
constructor, then the sequence assigned to its **pathway** argument is added to the specified `Composition`
using its `add_linear_processing_pathway <Composition.add_linear_processing_pathway>` method or its
`add_linear_learning_pathway <Composition.add_linear_processing_pathway>` method, depending on the
specifiation in the *pathway* argument (see those methods for corresponding specifications).  In this case,
the Pathway object returned by the constructor is the same as the one added to the Composition.

.. _Pathway_Template:

*Pathway as a Template*
~~~~~~~~~~~~~~~~~~~~~~~

If the **composition** argument is *not* specified in the Pathway's constructor, then
the Pathway is created on its own.  This can serve as a template for a Pathway assigned to a `Composition`,
by using it in the **processing_pathways** or **learning_pathways** argument of the constructor for a
Composition, or in its `add_linear_processing_pathway <Composition.add_linear_processing_pathway>` or
`add_linear_learning_pathway <Composition.add_linear_processing_pathway>` methods.  In any of these cases,
a new Pathway object is created and assigned to the Composition, and the template remains unassigned.

COMMENT:
*Roles*.  If the **roles** agument of the Pathway's constructor is specified, then the `NodeRole(s) <NodeRole>`
corresponding to the specified `PathwayRoles <PathwayRole>` are assigned to the nodes in the sequence
specified in the **pathways** argument.
COMMENT

.. _Pathway_Specification:

*Specification of* **pathway** *argument*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following formats can be used to specify a Pathway in the **pathway** argument of the constructor for the
Pathway, a `Composition`, or any of the Composition's methods used to add a Pathway to it.

    * `Node <Composition_Nodes>`: -- assigns the Node to a `SINGLETON` Pathway.
    ..
    .. _Pathway_Specification_List:

    * **list**: [`Node <Composition_Nodes>`, <`Projection <Projection>`,> `Node <Composition_Nodes>`...] --
      each item of the list must be a node (a `Mechanism <Mechanism>`, `Composition <Composition>` or a
      (Mechanism, `NodeRoles <NodeRole>`) tuple) or, optionally, a `Projection specification
      <Projection_Specification>` interposed between a pair of nodes.  The list must begin and end with a node.
    ..
    * **2-item tuple**: (Pathway, `LearningFunction`) -- used to specify a `learning Pathway
      <Composition_Learning_Pathways>`;  the 1st item must be a `Node <Composition_Nodes>` or list, as
      described above, and the 2nd item be a subclass of `LearningFunction`.

Structure
---------

.. _Pathway_Attribute:

A Pathway has the following primary attributes:

* `pathway <Pathway.pathway>` - if the Pathway was created on its own, this contains the specification provided in
  the **pathway** arg of its constructor; that is, depending upon how it was specified, it may or may not contain
  fully constructed `Components <Component>`.  This is passed to the **pathway** argument of a Composition's
  constructor or one of its `pathway creation methods <Composition_Pathway_Methods>` when the Pathway is used in the
  specifiation of any of these.  In contrast, when a Pathway is created by a Composition (and assigned to its
  `pathways <Composition.pathways>` attribute), then the actual `Mechanism(s) <Mechanism>` and/or `Composition(s)`
  that comprise `Nodes <Composition_Nodes>`, and the `Projection(s) <Projection>` between them, are listed in the
  Pathway's `pathway <Pathway.pathway>` attribute.

* `composition <Pathway.composition>` - contains the `Composition` that created the Pathway and to which it belongs,
  or None if it is a ``template <Pathway_Template>` (i.e., was constructed on its own).

* `roles <Pathway.roles>` and `Node <Composition_Nodes>` attributes - if the Pathway was created by a Composition,
  the `roles <Pathway.roles>` attribute `this lists the `PathwayRoles <PathwayRole>` assigned to it by the Compositon
  that correspond to the `NodeRoles <NodeRole`> of its Components, and the `Nodes <Composition_Nodes>` with each of
  those `NodeRoles <NodeRole>` is assigned to a corresponding attribute on the Pathway.  If the Pathway does not belong
  to a Composition (i.e., it is a `template <Pathway_Template>`), then these attributes return None.

* `learning_function <Pathway.learning_function>` - the LearningFunction assigned to the Pathway if it is a
  `learning Pathway <Composition_Learning_Pathway>` that belongs to a Composition; otherwise it is None.

Execution
---------

A Pathway cannot be executed on its own.  Its Components are executed when the Composition to which it belongs is
executed, by default in the order in which they appear in the `pathway <Pathway.pathway>` attribute;  however, this
can be modified by `Conditions <Condition>` added to the Composition's `scheduler <Composition.scheduler>`.

Class Reference
---------------

"""
import warnings
from enum import Enum
import typecheck as tc

from psyneulink.core.components.functions.learningfunctions import LearningFunction
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.compositions.composition import Composition, CompositionError, NodeRole
from psyneulink.core.globals.keywords import \
    ANY, COMPARATOR_MECHANISM, CONTEXT, MAYBE, NODE, PROJECTION, TARGET_MECHANISM
from psyneulink.core.globals.registry import register_category


__all__ = [
    'Pathway', 'PathwayRegistry', 'PathwayRole'
]


PathwayRegistry= {}


def _is_pathway_entry_spec(entry, desired_type:tc.enum(NODE, PROJECTION, ANY)):
    """Test whether pathway entry is specified type (NODE or PROJECTION)"""
    from psyneulink.core.components.projections.projection import _is_projection_spec
    node_specs = (Mechanism, Composition)
    is_node = is_proj = False

    if desired_type in {NODE, ANY}:
        is_node = (isinstance(entry, node_specs)
                   or (isinstance(entry, tuple)
                       and isinstance(entry[0], node_specs)
                       and (isinstance(entry[1], NodeRole) or
                            (isinstance(entry[1], list) and all(isinstance(nr, NodeRole) for nr in entry[1])))))

    if desired_type in {PROJECTION, ANY}:
        is_proj = (_is_projection_spec(entry)
                   or (isinstance(entry, tuple)
                       and _is_projection_spec(entry[0])
                       and entry[1] in {True, False, MAYBE}))

    if is_node or is_proj:
        return True
    else:
        return False


def _is_node_spec(value):
    return _is_pathway_entry_spec(value, NODE)


class PathwayRole(Enum):
    """

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

    CONTROL
        A `Pathway` that constitutes a `control pathway <Composition_Control_Pathways>` of the `Composition`.

    LEARNING
        A `Pathway` that constitutes a `learning sequence <Composition_Learning_Pathways>` of the `Composition`.

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
    Pathway(
        pathway,
        name=None,
        )

    A sequence of `Nodes <Composition_Nodes>` and `Projections <Projection>` in a `Composition`, or a template
    for one that can be assigned to one or more Compositions.

    Arguments
    ---------

    pathway : list[`Node <Composition_Nodes>`, <`Projection <Projection>`,> `Node <Composition_Nodes>`...]
        specifies list of `Nodes <Composition_Node>` and intercolated `Projections <Projection>` to be
        created for the Pathway.

    name : str : default see `name <Pathway.name>`
        specifies the name of the Pathway.

    Attributes
    ----------

    pathway : `Node <Component_Nodes>`, list, tuple, or dict.
        if the Pathway is created on its own, this contains the specification provided to the **pathway** argument
        of its constructor, and take any of the forms permitted for `Pathway specification <Pathway_Specification>`;
        if the Pathway is created by a Composition, this is a list of the `Nodes <Pathway_Nodes>` and intercolated
        `Projections <Projection>` in the Pathway (see `above <Pathway_Attribute>` for additional details).

    composition : `Composition` or None
        `Composition` to which the Pathway belongs;  if None, then Pathway is a `template <Pathway_Template>`.

    roles : list[`PathwayRole`]
        list of `PathwayRole(s) <PathwayRole>` assigned to the Pathway, based on the `NodeRole(s) <NodeRole>`
        assigned to its `Nodes <Composition>` in the `composition <Pathway.composition>` to which it belongs.

    learning_function : `LearningFunction` or None
        `LearningFunction` used by `LearningMechanism(s) <LearningMechanism>` associated with Pathway if
        it is a `learning pathway <Composition_Learning_Pathways>`.

    input : `Mechanism <Mechanism>` or None
        `INPUT` node if Pathway contains one.

    output : `Mechanism <Mechanism>` or None
        `OUTPUT` node if Pathway contains one.

    target : `Mechanism <Mechanism>` or None
        `TARGET` node if if Pathway contains one; same as `learning_components
        <Pathway.learning_components>`\\[*TARGET_MECHANISM*].

    comparator : `Mechanism <Mechanism>` or None
        `COMPARATOR_MECHANISM` if Pathway contains one; same as `learning_components
        <Pathway.learning_components>`\\[*COMPATOR_MECHANISM*].

    learning_components : dict
        dict containing the following entries if the Pathway is a `learning Pathway <Composition_Learning_Pathways>`
        (and is assigned `PathwayRole.LEARNING` in `roles <Pathway.roles>`):

          *TARGET_MECHANISM*: `ProcessingMechanism` (assigned to `target <Pathway.target>`)
          ..
          *COMPARATOR_MECHANISM*: `ComparatorMechanism` (assigned to `comparator <Pathway.comparator>`)
          ..
          *LEARNING_MECHANISMS*: `LearningMechanism` or list[`LearningMechanism`]
          ..
          *LEARNED_PROJECTIONS*: `Projection <Projection>` or list[`Projections <Projection>`]

        These are generated automatically and added to the `Composition` when the Pathway is assigned to it.

    name : str
        the name of the Pathway; if it is not specified in the **name** argument of the constructor, a
        default is assigned by PathwayRegistry (see `Naming` for conventions used for default and duplicate names).

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
                              f" {self.__class__.__name__}; it is assigned when the {self.__class__.__name__} "
                              f"is added to a {Composition.__name__}.")
        # composition arg must be a Composition
        if self.composition and not isinstance(self.composition, Composition):
            raise CompositionError(f"'composition' arg of constructor for {self.__class__.__name__} "
                              f"must be a {Composition.__name__}.")

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
        self.learning_components = {}
        self.pathway = pathway
        self.roles = set()

    def _assign_roles(self, composition):
        """Assign `PathwayRoles <PathwayRole>` to Pathway based `NodeRoles <NodeRole>` assigned to its `Nodes
        <Composition_Nodes>` by the **composition** to which it belongs.
        """
        assert composition, f'_assign_roles() cannot be called for {self.name} ' \
                            f'because it has not been assigned to a {Composition.__class__.__name__}.'
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
        if PathwayRole.INPUT in self.roles:
            input_node = next(n for n in self.pathway if n in self.composition.get_nodes_by_role(NodeRole.INPUT))
            if input_node:
                return input_node
            else:
                assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                              f"is assigned PathwayRole.INPUT but has no INPUT node."

    @property
    def output(self):
        if PathwayRole.OUTPUT in self.roles:
            output_node = next(n for n in self.pathway if n in self.composition.get_nodes_by_role(NodeRole.OUTPUT))
            if output_node:
                return output_node
            else:
                assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                              f"is assigned PathwayRole.OUTPUT but has no OUTPUT node."

    @property
    def target(self):
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
    def comparator(self):
        try:
            return self.learning_components[COMPARATOR_MECHANISM]
        except:
            if PathwayRole.LEARNING not in self.roles:
                warnings.warn(f"{self.__class__.__name__} {self.name} 'comparator' attribute "
                              f"is None because it is not a learning_pathway.")
            else:
                assert False, f"PROGRAM ERROR: {self.__class__.__name__} {self.name} of {self.composition.name} " \
                              f"has PathwayRole.LEARNING assigned but no 'comparator' attribute."
            return None
