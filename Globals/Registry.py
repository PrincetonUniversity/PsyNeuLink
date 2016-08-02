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
import Functions.Function
from collections import namedtuple
from inspect import isclass

# IMPLEMENTATION NOTE:
# - Implement Registry as class, and each Registry as subclass
# - Implement RegistryPreferenceSet as PreferenceSet subclass, and assign prefs attribute to each Registry object

DEFAULT_REGISTRY_VERBOSITY = False
from Globals.Keywords import *
RegistryVerbosePrefs = {
    kwPreferenceSet: DEFAULT_REGISTRY_VERBOSITY,
    kwFunctionPreferenceSet: DEFAULT_REGISTRY_VERBOSITY,
    kwProcessFunctionCategory: DEFAULT_REGISTRY_VERBOSITY,
    kwMechanismFunctionCategory: DEFAULT_REGISTRY_VERBOSITY,
    kwStateFunctionCategory: DEFAULT_REGISTRY_VERBOSITY,
    kwInputState: DEFAULT_REGISTRY_VERBOSITY,
    kwMechanismParameterState: DEFAULT_REGISTRY_VERBOSITY,
    kwOutputState: DEFAULT_REGISTRY_VERBOSITY,
    kwDDM: DEFAULT_REGISTRY_VERBOSITY,
    kwProjectionFunctionCategory: DEFAULT_REGISTRY_VERBOSITY,
    kwControlSignal: DEFAULT_REGISTRY_VERBOSITY,
    kwMapping: DEFAULT_REGISTRY_VERBOSITY,
    kwUtilityFunctionCategory: DEFAULT_REGISTRY_VERBOSITY,
}

RegistryEntry = namedtuple('RegistryTuple', 'subclass, instanceDict, instanceCount, default')

def rreplace(myStr, old, new, count):
    return myStr[::-1].replace(old[::-1], new[::-1], count)[::-1]


class RegistryError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def register_category(entry, base_class, registry=NotImplemented, context='Registry'):
# DOCUMENT:
    """Maintains registry of subclasses for base_class, names instances incrementally, and sets default

    Arguments:
    - entry (object or class)
    - base_class (parent class for entry)
    - registry (dict)

# DOCUMENTATION:
             - Naming procedure / conventions
             - Default procedure /conventions

# IMPLEMENTATION NOTE:
        ADD DEFAULT MANAGEMENT (USEFUL AT LEAST FOR PROCESS... OTHERS ARE CONTEXT-SPECIFIC)
        # # MechanismRegistry ------------------------------------------------------------------------
        # #
        # # Dictionary of registered Mechanism subclasses; each entry has:
        # #     - key: Mechanism subclass name (functionType)
        # #     - value: MechanismEntry tuple (mechanism, instanceCount, default)
        # #              Notes:
        # #              * instanceCount is incremented each time a new default instance is created
        # #              * only one default is allowed;  if a mechanism registers itself as default,
        # #                  that displaces whatever was the default previously;  initially it is DDM
        #
        # # MechanismRegistry = {DefaultReceiver.name:(DefaultReceiverMechanism, 1)}
        #

    :param entry:
    :param default:
    :return:
    """

    if not issubclass(base_class, object):
        raise RegistryError("base_class ({0}) for registry must be a subclass of Function".format(base_class))

    # if registry is NotImplemented:
    #     try:
    #         registry = base_class.registry
    #     except AttributeError:
    #         raise RegistryError("{0} must be a dict".format(registry))

    if not isinstance(registry, dict):
        raise RegistryError("Registry ({0}) for {1} must be a dict".format(registry,base_class.__name__))

    # If entry is instance of the subclass:
    if isinstance(entry, base_class):

        subclass_name = entry.__class__.__name__

        # If subclass is registered (i.e., there is an entry for subclass_name), then:
        if subclass_name in registry:

            # Get and increment instanceCount
            instanceCount = registry[subclass_name].instanceCount + 1

            # If instance does not have a name, set instance's name to "subclass_name-1"
            if entry.name is NotImplemented:
                entry.name = subclass_name+'-1'

            # Check for instance name in instanceDict for its subclass;
            # - if name exists, add numerical suffix if none, and increment if already present
            old_entry_name = entry.name
            while entry.name in registry[subclass_name].instanceDict:
                try:
                    # Check if name ends in '-number'
                    numerical_suffix = [int(s) for s in entry.name.rsplit('-') if s.isdigit()][-1]
                except IndexError:
                    # Otherwise, add '-1' as suffix
                    entry.name = entry.name+'-1'
                else:
                # If so, replace only final occurence of '-number' with '-number+1'
                    if numerical_suffix:
                        # entry.name.rreplace('-'+str(numerical_suffix),'-'+str(numerical_suffix+1),1)
                        entry.name = rreplace(entry.name, '-'+str(numerical_suffix),'-'+str(numerical_suffix+1),1)
                        if RegistryVerbosePrefs[base_class.__name__]:
                            print("Object named {0} already registered; current one will be re-named {1}.".
                                  format(old_entry_name, entry.name))

            # Add instance to instanceDict:
            registry[subclass_name].instanceDict.update({entry.name: entry})

            # Update instanceCount in registry:
            registry[subclass_name] = registry[subclass_name]._replace(instanceCount=instanceCount)

        # If subclass is not already registered in registry, then:
        else:
            # Set instance's name to first instance:
            entry.name = entry.name+"-1"

            # Create instance dict:
            instanceDict = {entry.name: entry}

            # Register subclass with instance count of 1:
            registry[subclass_name] = RegistryEntry(type(entry), instanceDict, 1, False)


    # If entry is a reference to the subclass (rather than an instance of it)
    elif issubclass(entry, base_class):
        subclass_name = entry.__name__
        # If it is already there, ignore
        if subclass_name in registry:
            pass
        # If it is not there enter in registry but instantiate empty instanceDict and set instance count = 0
        else:
            registry[subclass_name] = RegistryEntry(entry, {}, 0, False)

    else:
        raise RegistryError("Requested entry {0} not of type {1}".format(entry, base_class))



# def set_default_mechanism(mechanism_subclass):
#     """Sets DefaultMechanism to specified subclass
#
#     :param mechanism_subclass:
#     :return:
#     """
#
#     if not (issubclass(mechanism_subclass, Mechanism)):
#         raise MechanismError("Requested mechanism {0} not of type {1}".format(mechanism_subclass, type(Mechanism)))
#
#     # Remove existing default flag
#     old_default_name = NotImplemented
#     for subclass_name in MechanismRegistry:
#         if MechanismRegistry[subclass_name].default:
#             old_default_name = subclass_name
#             MechanismRegistry[subclass_name] = MechanismRegistry[subclass_name]._replace(default=False)
#
#
#     # Flag specified subclass as default
#     try:
#         MechanismRegistry[mechanism_subclass.functionType] =\
#             MechanismRegistry[mechanism_subclass.functionType]._replace(default=True)
#     # Not yet registered, so do so as default
#     except KeyError:
#         register_mechanism_subclass(mechanism_subclass)
#         MechanismRegistry[mechanism_subclass.functionType] =\
#             MechanismRegistry[mechanism_subclass.functionType]._replace(default=True)

#     # Assign to DefaultMechanism
#     Functions.DefaultMechanism = MechanismRegistry[mechanism_subclass.name].mechanismSubclass
# mechanism_subclass
#     # Issue warning
#     if self.prefs.verbosePref:
#         print("{0} set as new default mechanism ({1}) removed)".format(mechanism_subclass.name, old_default_name))

