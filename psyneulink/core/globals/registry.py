# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Registry ************************************************************

import inspect
import re

from collections import defaultdict, namedtuple

from psyneulink.core.globals.keywords import \
    CONTROL_PROJECTION, DDM_MECHANISM, GATING_SIGNAL, INPUT_PORT, MAPPING_PROJECTION, OUTPUT_PORT, \
    FUNCTION_COMPONENT_CATEGORY, COMPONENT_PREFERENCE_SET, MECHANISM_COMPONENT_CATEGORY, \
    PARAMETER_PORT, PREFERENCE_SET, PROCESS_COMPONENT_CATEGORY, PROJECTION_COMPONENT_CATEGORY, \
    PORT_COMPONENT_CATEGORY

__all__ = [
    'RegistryError',
    'clear_registry',
    'process_registry_object_instances'
]

# IMPLEMENTATION NOTE:
# - Implement Registry as class, and each Registry as subclass
# - Implement RegistryPreferenceSet as PreferenceSet subclass, and assign prefs attribute to each Registry object

DEFAULT_REGISTRY_VERBOSITY = False
RegistryVerbosePrefs = {
    PREFERENCE_SET: DEFAULT_REGISTRY_VERBOSITY,
    COMPONENT_PREFERENCE_SET: DEFAULT_REGISTRY_VERBOSITY,
    MECHANISM_COMPONENT_CATEGORY: DEFAULT_REGISTRY_VERBOSITY,
    PORT_COMPONENT_CATEGORY: DEFAULT_REGISTRY_VERBOSITY,
    INPUT_PORT: DEFAULT_REGISTRY_VERBOSITY,
    PARAMETER_PORT: DEFAULT_REGISTRY_VERBOSITY,
    OUTPUT_PORT: DEFAULT_REGISTRY_VERBOSITY,
    GATING_SIGNAL: DEFAULT_REGISTRY_VERBOSITY,
    DDM_MECHANISM: DEFAULT_REGISTRY_VERBOSITY,
    PROJECTION_COMPONENT_CATEGORY: DEFAULT_REGISTRY_VERBOSITY,
    CONTROL_PROJECTION: DEFAULT_REGISTRY_VERBOSITY,
    MAPPING_PROJECTION: DEFAULT_REGISTRY_VERBOSITY,
    FUNCTION_COMPONENT_CATEGORY: DEFAULT_REGISTRY_VERBOSITY,
}

RegistryEntry = namedtuple('RegistryTuple', 'subclass, instanceDict, instanceCount, renamed_instance_counts, default')

numeric_suffix_pat = re.compile(r'(.*)-\d+$')


class RegistryError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def register_category(entry,
                      base_class,
                      name=None,
                      registry=None,
                      context='Registry'):
    """Create a category within the specified registry.

    Arguments
    ---------

    base_class : Class
        PsyNeuLink Base class of Category to be added to Registry.

    name : str
        PsyNeuLink Base class of Category for which Registry is to be created.

    Registry : Dict
        Registry to which Category should be added.

    context : Context or str

    COMMENT:
    - entry (object or class)
    - base_class (parent class for entry)
    - registry (dict)


    # DOCUMENTATION:
             - Naming procedure / conventions
             - Default procedure /conventions

             TBI:
             - sub group option:  allows instances to be sub-grouped by the attribute passed in sub_group_attr arg

             If sub_group_attr, then:
                 - instead of implementing instance dict directly, implement subgroup dict
                 - when an entry is made, check for owner in sub-dict
                 - if no owner, create entry for owner, instance dict for that owner, and log instance in instance dict
                 - if owner is found, but no instances, create instance dict for that owner
                 - if owner and instance dict exist, make entry

                 - implement instance dict within subgroup



    # IMPLEMENTATION NOTE:
        ADD DEFAULT MANAGEMENT (USEFUL AT LEAST FOR PROCESS... OTHERS ARE CONTEXT-SPECIFIC)
        # # MechanismRegistry ------------------------------------------------------------------------
        # #
        # # Dictionary of registered Mechanism subclasses; each entry has:
        # #     - key: Mechanism component type name (componentType)
        # #     - value: MechanismEntry tuple (Mechanism, instanceCount, default)
        # #              Notes:
        # #              * instanceCount is incremented each time a new default instance is created
        # #              * only one default is allowed;  if a Mechanism registers itself as default,
        # #                  that displaces whatever was the default previously;  initially it is DDM
        #
        # # MechanismRegistry = {DefaultReceiver.name:(DefaultReceiverMechanism, 1)}
        #
    COMMENT
    """

    # IMPLEMENTATION NOTE:  Move to Port when that is implemented as ABC
    from psyneulink.core.components.shellclasses import Port
    if inspect.isclass(entry) and issubclass(entry, Port):
        try:
            entry.portAttributes
        except AttributeError:
            raise RegistryError("PROGRAM ERROR: {} must implement a stateSpecificParams attribute".
                                format(entry.__name__))
        try:
            entry.connectsWith
        except AttributeError:
            raise RegistryError("PROGRAM ERROR: {} must implement a connectsWith attribute".format(entry.__name__))
        try:
            entry.connectsWithAttribute
        except AttributeError:
            raise RegistryError("PROGRAM ERROR: {} must implement a connectsWithAttribute attribute".
                                format(entry.__name__))
        try:
            entry.projectionSocket
        except AttributeError:
            raise RegistryError("PROGRAM ERROR: {} must implement a projectionSocket attribute".format(entry.__name__))
        try:
            entry.modulators
        except AttributeError:
            raise RegistryError("PROGRAM ERROR: {} must implement a modulators attribute".format(entry.__name__))

    if not isinstance(registry, dict):
        raise RegistryError("Registry ({0}) for {1} must be a dict".format(registry,base_class.__name__))

    # if sub_group_attr:
    #     if not isinstance(sub_group_attr, str):
    #         raise RegistryError("sub_group_attr arg ({0}) must be a str that is the name of an attribute of {1} ".
    #                             format(sub_group_attr,entry.__class__.__name__))
    #     try:
    #         sub_group = getattr(entry,sub_group_attr)
    #     except AttributeError:
    #         raise RegistryError("sub_group_attr arg ({0}) must be an attribute of {1} ".
    #                             format(sub_group_attr,entry.__class__.__name__))

    # If entry is an instance (presumably of a component type of the base class):
    if isinstance(entry, base_class):

        try:
            component_type_name = entry.componentName
        except AttributeError:
            try:
                component_type_name = entry.componentType
            except AttributeError:
                component_type_name = entry.__class__.__name__

        # Component type is registered (i.e., there is an entry for component_type_name)
        if component_type_name in registry:
            register_instance(entry=entry,
                              name=name,
                              base_class=base_class,
                              registry=registry,
                              sub_dict=component_type_name)

        # If component type is not already registered in registry, then:
        else:
            # Set instance's name to first instance:
            # If name was not provided, assign component_type_name-1 as default;
            if name is None:
                try:
                    entry.name = component_type_name + "-0"
                except TypeError:
                    entry.name = entry.__class__.__name__

            else:
                entry.name = name

            # Create instance dict:
            instanceDict = {entry.name: entry}
            renamed_instance_counts = defaultdict(int)

            # Register component type with instance count of 1:
            registry[component_type_name] = RegistryEntry(type(entry), instanceDict, 1, renamed_instance_counts, False)


    # If entry is a reference to the component type (rather than an instance of it)
    elif issubclass(entry, base_class):
        component_type_name = entry.__name__
        # If it is already there, ignore
        if component_type_name in registry:
            pass
        # If it is not there:
        # - create entry for component type in registry
        # - instantiate empty instanceDict
        # - set instance count = 0
        else:
            registry[component_type_name] = RegistryEntry(entry, {}, 0, defaultdict(int), False)

    else:
        raise RegistryError("Requested entry {0} not of type {1}".format(entry, base_class))


def register_instance(entry, name, base_class, registry, sub_dict):

    renamed_instance_counts = registry[sub_dict].renamed_instance_counts
    renamed = False
    # If entry (instance) name is None, set entry's name to sub_dict-n where n is the next available numeric suffix
    # (starting at 0) based on the number of unnamed/renamed sub_dict objects that have already been assigned names
    if name is None:
        entry.name = '{0}-{1}'.format(sub_dict, renamed_instance_counts[sub_dict])
        renamed = True
    else:
        entry.name = name

    while entry.name in registry[sub_dict].instanceDict:
        # if the decided name (provided or determined) is already assigned to an object, get the non-suffixed name,
        # and append the proper new suffix according to the number of objects that have been assigned that name
        # NOTE: the while is to handle a scenario in which a user specifies a name that uses our convention but
        #   does not follow our pattern. In this case, we will produce a unique name, but the "count" will be off
        #   e.g. user gives a mechanism the name TransferMechanism-5, then the fifth unnamed TransferMechanism
        #   will be named TransferMechanism-6
        # TODO: an ambiguation problem - is the name "MappingProjection x to y-1"
        # the second projection from x to y, or the first projection from x to y-1?
        match = numeric_suffix_pat.match(entry.name)

        if match is None:
            renamed_instance_counts[entry.name] += 1
            entry.name += '-{0}'.format(renamed_instance_counts[entry.name])
        else:
            name_stripped_of_suffix = match.groups()[0]

            if name_stripped_of_suffix in renamed_instance_counts:
                # try to detect unsuffixed version first as base name
                renamed_instance_counts[name_stripped_of_suffix] += 1
                entry.name = numeric_suffix_pat.sub(
                    r'\1-{0}'.format(
                        renamed_instance_counts[name_stripped_of_suffix]
                    ),
                    entry.name
                )
            else:
                # the second time a -\d suffixed name (not renamed though) appears
                renamed_instance_counts[entry.name] += 1
                entry.name += '-{0}'.format(renamed_instance_counts[entry.name])

    # Add instance to instanceDict:
    registry[sub_dict].instanceDict.update({entry.name: entry})

    # Update instanceCount in registry:
    registry[sub_dict] = registry[sub_dict]._replace(instanceCount=registry[sub_dict].instanceCount + 1)

    # increment the base (non-suffixed) name count
    if renamed:
        match = numeric_suffix_pat.match(entry.name)
        if match is None:
            renamed_instance_counts[entry.name] += 1
        else:
            renamed_instance_counts[match.groups()[0]] += 1

def rename_instance_in_registry(registry, category, name=None, component=None):
    """Rename instance in category registry

    Instance to be renamed can be specified by a reference to the component or its name.
    COMMENT:
    DEPRECACTED (SEE IMPLEMENTATION NOTE BELOW)
    If the name of the instance was a default name, and it was the last in the sequence,
        decrement renamed_instance_counts and if it was the only one, remove that name from the renamed_instance list
    COMMENT
    """

    registry_entry = registry[category]

    if not (name or component):
        raise RegistryError("Must specify a name or component to remove an entry of {}".
                            format(registry.__class__.__name__))
    if (name and component) and name != component.name:
        raise RegistryError("Conflicting  name ({}) and component ({}) specified for entry to remove from {}".
                            format(name, component.name, registry.__class__.__name__))
    if component and not name:
        for n, c in registry_entry.instanceDict.items():
            if component == c:
                name = n

    try:
        clear_registry(registry_entry.instanceDict[name]._portRegistry)
    except (AttributeError):
        pass

    # Delete instance
    del registry_entry.instanceDict[name]

    # Decrement count for instances in entry
    instance_count = registry_entry.instanceCount - 1

    # IMPLEMENTATION NOTE:
    #    Don't decrement renamed_instance_counts as:
    #        - doing so would require checking that the item being removed is the last in the sequence
    #          (to avoid fouling subsequent indexing);
    #        - it might be confusing for a subsequently added item to have the same name as one previously removed.
    # # If instance's name was a duplicate with appended index, decrement the count for that item (and remove if it is 0)
    # for base_name, count in registry_entry.renamed_instance_counts.items():
    #     if base_name in name:
    #         registry_entry.renamed_instance_counts[base_name] -= 1
    #         if registry_entry.renamed_instance_counts[base_name] == 0:
    #             del registry_entry.renamed_instance_counts[base_name]
    #         break
    # Reassign entry with new values
    registry[category] = RegistryEntry(registry_entry.subclass,
                                       registry_entry.instanceDict,
                                       instance_count,
                                       registry_entry.renamed_instance_counts,
                                       registry_entry.default)

def remove_instance_from_registry(registry, category, name=None, component=None):
    """Remove instance from registry category entry

    Instance to be removed can be specified by a reference to the component or its name.
    Instance count for the category is decremented
    If the name of the instance was a default name, and it was the last in the sequence,
        decrement renamed_instance_counts and if it was the only one, remove that name from the renamed_instance list
    """

    registry_entry = registry[category]

    if not (name or component):
        raise RegistryError("Must specify a name or component to remove an entry of {}".
                            format(registry.__class__.__name__))
    if (name and component) and name != component.name:
        raise RegistryError("Conflicting  name ({}) and component ({}) specified for entry to remove from {}".
                            format(name, component.name, registry.__class__.__name__))
    if component and not name:
        for n, c in registry_entry.instanceDict.items():
            if component == c:
                name = n

    try:
        clear_registry(registry_entry.instanceDict[name]._portRegistry)
    except (AttributeError):
        pass

    # Delete instance
    del registry_entry.instanceDict[name]

    # Decrement count for instances in entry
    instance_count = registry_entry.instanceCount - 1

    # IMPLEMENTATION NOTE:
    #    Don't decrement renamed_instance_counts as:
    #        - doing so would require checking that the item being removed is the last in the sequence
    #          (to avoid fouling subsequent indexing);
    #        - it might be confusing for a subsequently added item to have the same name as one previously removed.
    # # If instance's name was a duplicate with appended index, decrement the count for that item (and remove if it is 0)
    # for base_name, count in registry_entry.renamed_instance_counts.items():
    #     if base_name in name:
    #         registry_entry.renamed_instance_counts[base_name] -= 1
    #         if registry_entry.renamed_instance_counts[base_name] == 0:
    #             del registry_entry.renamed_instance_counts[base_name]
    #         break
    # Reassign entry with new values
    registry[category] = RegistryEntry(registry_entry.subclass,
                                       registry_entry.instanceDict,
                                       instance_count,
                                       registry_entry.renamed_instance_counts,
                                       registry_entry.default)

def clear_registry(registry):
    """Clear specified registry of all entries, but leave any categories created within it intact.

    .. note::
       This method should be used with caution.  It is used primarily in unit tests, to insure consistency of naming
       within a given test.  Calling it outside of testing may allow new Components of the same type to be created with
       exactly the same PsyNeuLink name as exsiting ones within the same Python namespace.

    """
    for category in registry:
        instance_dict = registry[category].instanceDict.copy()
        for name in instance_dict:
            remove_instance_from_registry(registry, category, name)
        registry[category].renamed_instance_counts.clear()

def process_registry_object_instances(registry, func):
    for category in registry:
        for (name, obj) in registry[category].instanceDict.items():
            func(name, obj)
