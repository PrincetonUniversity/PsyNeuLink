# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************** Pathway **************************************************************

import warnings

import typecheck as tc

from psyneulink.core.components.functions.learningfunctions import LearningFunction
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.compositions.composition import Composition, CompositionError
from psyneulink.core.globals.keywords import ANY, COMPARATOR_MECHANISM, MAYBE, NODE, PROJECTION, TARGET_MECHANISM
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import NodeRole, PathwayRole, convert_to_list

__all__ = [
    'Pathway', 'PathwayRegistry'
]


def _is_pathway_entry_spec(entry, desired_type:tc.enum(NODE, PROJECTION, ANY)):
    """Test whether pathway entry is specified type (NODE or PROJECTION)"""
    from psyneulink.core.components.projections.projection import _is_projection_spec
    node_specs = (Mechanism, Composition)
    is_node = is_proj = False

    if desired_type in {NODE, ANY}:
        is_node = (isinstance(entry, node_specs)
                   or (isinstance(entry, tuple)
                       and isinstance(entry[0], node_specs)
                       and isinstance(entry[1], NodeRole)))

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


PathwayRegistry= {}


class Pathway(object):
    """
        A sequence of `Nodes <Composition_Nodes>` and `Projections <Projection>` in a `Composition`, or a template
        for one that can be assigned to one or more Compositions.

        **Creating a Pathway**
        ----------------------

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

        *Roles*.  If the **roles** agument of the Pathway's constructor is specified, then the `NodeRole(s) <NodeRole>`
        corresponding to the specified `PathwayRoles <PathwayRole>` are assigned to the nodes in the sequence
        specified in the **pathways** argument.

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

        Arguments
        ---------

        pathway : list[`Node <Composition_Nodes>`, <`Projection <Projection>`,> `Node <Composition_Nodes>`...]
            specifies list of `Nodes <Composition_Node>` and intercolated `Projections <Projection>` to be
            created for the Pathway.

        composition : `Composition` default None
            specifies `Composition` to which the Pathway should be assigned.

        name : str : default see `name <Pathway.name>`
            specifies the name of the Pathway.

        Attributes
        ----------

        pathway : list[`Node <Pathway_Nodes>`, `Projection <Projection>`, `Node <Pathway_Nodes>`...]
            list of `Nodes <Pathway_Nodes>` and intercolated `Projections <Projection>` in the Pathway.

        composition : `Composition` or None
            `Composition` to which the Pathway belongs;  if None, then Pathway is `template <Pathway_Template>`.

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
            composition:Composition=None,
            name=None,
            context=None
    ):

        # If called from command line, being used as a template, so don't register
        if context.source == ContextFlags.COMMAND_LINE:
            # But do pass through name so that it can be used to construct the instance that will be used
            self.name = name
        else:
            # Sets name
            register_category(
                entry=self,
                base_class=Pathway,
                registry=PathwayRegistry,
                name=name
            )
        self.composition = composition
        self.learning_components = {}

        # # Check validity of pathway
        # for i in range(0, len(pathway)):
        #     # Odd items must be a node (Mechanism or Composition)
        #     if not i % 2:
        #         if not isinstance(pathway[i], (Mechanism, Composition)):
        #             raise CompositionError(f"Item {i} of {self.name} ({pathway[0]}) must be a node "
        #                                    f"({Mechanism.__name__} or {Composition.__name__}).")
        #     # Even items must be a Projection
        #     elif not isinstance(pathway[i], Projection):
        #             raise CompositionError(f"Item {i} of {self.name} ({pathway[0]}) must be a `{Projection.__name}.")
        # # If len is not odd, then must be missing a node
        # if not len(pathway) % 2:
        #     raise CompositionError(f"'pathway' arg of {self.name} is missing a terminal node.")

        self.pathway = pathway

        # # roles = set(convert_to_list(roles or []))
        # # for role in roles:
        # #     if not isinstance(role, PathwayRole):
        # #         raise CompositionError(f"Item ({role}) in 'roles' arg of {self.__class__.__name__} {self.name} "
        # #                                f"is not a {PathwayRole.__name__}.")
        # self.roles = roles
        self.roles = set()

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
