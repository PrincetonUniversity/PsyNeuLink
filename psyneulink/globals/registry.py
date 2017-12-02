# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Registry ************************************************************
#

import re

from collections import namedtuple

from psyneulink.globals.keywords import CONTROL_PROJECTION, DDM_MECHANISM, GATING_SIGNAL, INPUT_STATE, MAPPING_PROJECTION, OUTPUT_STATE, PARAMETER_STATE, kwComponentCategory, kwComponentPreferenceSet, kwMechanismComponentCategory, kwPreferenceSet, kwProcessComponentCategory, kwProjectionComponentCategory, kwStateComponentCategory, kwSystemComponentCategory

__all__ = [
    'RegistryError',
    'clear_registry'
]

# IMPLEMENTATION NOTE:
# - Implement Registry as class, and each Registry as subclass
# - Implement RegistryPreferenceSet as PreferenceSet subclass, and assign prefs attribute to each Registry object

DEFAULT_REGISTRY_VERBOSITY = False
RegistryVerbosePrefs = {
    kwPreferenceSet: DEFAULT_REGISTRY_VERBOSITY,
    kwComponentPreferenceSet: DEFAULT_REGISTRY_VERBOSITY,
    kwSystemComponentCategory: DEFAULT_REGISTRY_VERBOSITY,
    kwProcessComponentCategory: DEFAULT_REGISTRY_VERBOSITY,
    kwMechanismComponentCategory: DEFAULT_REGISTRY_VERBOSITY,
    kwStateComponentCategory: DEFAULT_REGISTRY_VERBOSITY,
    INPUT_STATE: DEFAULT_REGISTRY_VERBOSITY,
    PARAMETER_STATE: DEFAULT_REGISTRY_VERBOSITY,
    OUTPUT_STATE: DEFAULT_REGISTRY_VERBOSITY,
    GATING_SIGNAL: DEFAULT_REGISTRY_VERBOSITY,
    DDM_MECHANISM: DEFAULT_REGISTRY_VERBOSITY,
    kwProjectionComponentCategory: DEFAULT_REGISTRY_VERBOSITY,
    CONTROL_PROJECTION: DEFAULT_REGISTRY_VERBOSITY,
    MAPPING_PROJECTION: DEFAULT_REGISTRY_VERBOSITY,
    kwComponentCategory: DEFAULT_REGISTRY_VERBOSITY,
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
# DOCUMENT:
    """Maintains registry of subclasses for base_class, names instances incrementally (if duplicates), and sets default

    Arguments:
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

    :param entry:
    :param default:
    :return:
    """

    # IMPLEMENTATION NOTE:  Move to State when that is implemented as ABC
    import inspect
    from psyneulink.components.states.state import State, State_Base
    if inspect.isclass(entry) and issubclass(entry, State) and not entry == State_Base:
        try:
           entry.stateAttributes
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


    from psyneulink.components.component import Component
    from psyneulink.globals.preferences.preferenceset import PreferenceSet
    if not issubclass(base_class, (Component, PreferenceSet)):
        raise RegistryError("base_class ({0}) for registry must be a subclass of "
                            "Component or PreferenceSet".format(base_class))

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
            if not name:
                entry.name = component_type_name + "-0"
            else:
                entry.name = name

            # Create instance dict:
            instanceDict = {entry.name: entry}
            if name is None:
                renamed_instance_counts = {component_type_name: 1}
            else:
                renamed_instance_counts = {component_type_name: 0}

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
            registry[component_type_name] = RegistryEntry(entry, {}, 0, {component_type_name: 0}, False)

    else:
        raise RegistryError("Requested entry {0} not of type {1}".format(entry, base_class))


def register_instance(entry, name, base_class, registry, sub_dict):

    renamed_instance_counts = registry[sub_dict].renamed_instance_counts

    # If entry (instance) name is None, set entry's name to sub_dict-n where n is the next available numeric suffix
    # (starting at 0) based on the number of unnamed/renamed sub_dict objects that have already been assigned names
    if not name:
        entry.name = '{0}-{1}'.format(sub_dict, renamed_instance_counts[sub_dict])
        renamed_instance_counts[sub_dict] += 1
    else:
        entry.name = name

    while entry.name in registry[sub_dict].instanceDict:
        # if the decided name (provided or determined) is already assigned to an object, get the non-suffixed name,
        # and append the proper new suffix according to the number of objects that have been assigned that name
        # NOTE: the while is to handle a scenario in which a user specifies a name that uses our convention but
        #   does not follow our pattern. In this case, we will produce a unique name, but the "count" will be off
        #   e.g. user gives a mechanism the name TransferMechanism-5, then the fifth unnamed TransferMechanism
        #   will be named TransferMechanism-6
        match = numeric_suffix_pat.match(entry.name)
        if not match:
            name_stripped_of_suffix = entry.name
            try:
                renamed_instance_counts[entry.name] += 1
            except KeyError:
                renamed_instance_counts[entry.name] = 1
            entry.name += '-{0}'.format(renamed_instance_counts[entry.name])
        else:
            name_stripped_of_suffix = match.groups()[0]
            entry.name = numeric_suffix_pat.sub(r'\1-{0}'.
                                                format(renamed_instance_counts[name_stripped_of_suffix]), entry.name)

    # Add instance to instanceDict:
    registry[sub_dict].instanceDict.update({entry.name: entry})

    # Update instanceCount in registry:
    registry[sub_dict] = registry[sub_dict]._replace(instanceCount=registry[sub_dict].instanceCount + 1)

def clear_registry(registry):
    registry.clear()
