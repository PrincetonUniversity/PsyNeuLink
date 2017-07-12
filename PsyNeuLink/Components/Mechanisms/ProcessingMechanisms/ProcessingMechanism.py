# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ProcessingMechanism ****************************************************

"""

Overview
--------

A ProcessingMechanism is a type of `Mechanism <Mechanism>` that transforms its input in some way.  A
ProcessingMechanism always receives its input either from another ProcessingMechanism, or from the input to a `process
<Process>` or `system <System>` when it is executed.  Similarly, its output is generally conveyed to another
ProcessingMechanism or used as the ouput for a process or system.  However, the output of a ProcessingMechanism may
also be used by an `AdaptiveMechanism` to modify the parameters of other components (or its own). ProcessingMechanisms
are always executed before all AdaptiveMechanisms in the Process and/or System to which they belong, so that any
modificatons made by the AdpativeMechanism are available to all ProcessingMechanisms in the next `TRIAL`.

.. _ProcessingMechanism_Creation:

Creating a ProcessingMechanism
------------------------------

A ProcessingMechanism can be created by using the standard Python method of calling the constructor for the desired
type. Some types of ProcessingMechanism (for example, `ObjectiveMechanisms <ObjectiveMechanism>`) are also created
when a System or Process is created, if `learning <LINK>` and/or `control <LINK>` have been specified for it.

.. _AdaptiveMechanism_Structure:

Structure
---------

A ProcessingMechanism has the same basic structure as a `Mechanism <Mechanisms>`.  See the documentation for
individual subtypes of ProcessingMechanism for more specific information about their structure.

.. _Comparator_Execution:

Execution
---------

A ProcessingMechanism always executes before any `AdaptiveMechanisms <AdaptiveMechanism>` in the process or
system to which it belongs.

"""

from PsyNeuLink.Components.Mechanisms.Mechanism import *
from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Globals.Keywords import *

# ControlMechanismRegistry = {}


class ProcessingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ProcessingMechanism_Base(Mechanism_Base):
    # DOCUMENTATION: this is a TYPE and subclasses are SUBTYPES
    #                primary purpose is to implement TYPE level preferences for all processing mechanisms
    #                inherits all attributes and methods of Mechanism -- see Mechanism for documentation
    # IMPLEMENT: consider moving any properties of processing mechanisms not used by control mechanisms to here
    """Abstract class for processing mechanism subclasses
   """

    componentType = "ProcessingMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ProcessingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per control_signal)
    variableClassDefault = defaultControlAllocation

    def __init__(self,
                 variable=None,
                 size=None,
                 input_states=None,
                 output_states=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Abstract class for processing mechanisms

        :param variable: (value)
        :param size: (int or list/array of ints)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        """

        self.system = None

        # region Fill in and infer variable and size if they aren't specified in args
        # if variable is None and size is None:
        #     variable = self.variableClassDefault
        # 6/30/17 now handled in the individual subclasses' __init__() methods because each subclass has different
        # expected behavior when variable is None and size is None.

        def checkAndCastInt(x):
            if not isinstance(x, numbers.Number):
                raise ProcessingMechanismError("An element ({}) in size is not a number.".format(x))
            if x < 1:
                raise ProcessingMechanismError("An element ({}) in size is not a positive number.".format(x))
            try:
                int_x = int(x)
            except:
                raise ProcessingMechanismError(
                    "Failed to convert an element ({}) in size argument to an integer. size "
                    "should be a number, or iterable of numbers, which are integers or "
                    "can be converted to integers.".format(x))
            if int_x != x:
                if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                    warnings.warn("When an element ({}) in the size argument was cast to "
                                  "integer, its value changed to {}.".format(x, int_x))
            return int_x

        # region Convert variable (if given) to a 2D array, and size (if given) to a 1D integer array
        try:
            if variable is not None:
                variable = np.atleast_2d(variable)
                if len(np.shape(variable)) > 2:  # number of dimensions of variable > 2
                    if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                        warnings.warn("variable had more than two dimensions (had {} dimensions) "
                                      "so only the first element of its second-highest-numbered axis will be"
                                      " used".format(len(np.shape(variable))))
                    while len(np.shape(variable)) > 2:  # reduce the dimensions of variable
                        variable = variable[0]

                # 6/30/17 (CW): Previously, using variable or default_input_value to create input states of differing
                # lengths (e.g. default_input_value = [[1, 2], [1, 2, 3]]) caused a bug. The if statement below
                # fixes this bug. This solution is ugly, though.
                if isinstance(variable[0], list) or isinstance(variable[0], np.ndarray):
                    allLists = True
                    for i in range(len(variable[0])):
                        if isinstance(variable[0][i], list) or isinstance(variable[0][i], np.ndarray):
                            variable[0][i] = np.array(variable[0][i])
                        else:
                            allLists = False
                            break
                    if allLists:
                        variable = variable[0]
        except:
            raise ProcessingMechanismError("Failed to convert variable (of type {})"
                                           " to a 2D array".format(type(variable)))

        try:
            if size is not None:
                size = np.atleast_1d(size)
                if len(np.shape(size)) > 1:  # number of dimensions of size > 1
                    if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                        warnings.warn("size had more than one dimension (size had {} dimensions), so only the first "
                                      "element of its highest-numbered axis will be used".format(len(np.shape(size))))
                    while len(np.shape(size)) > 1:  # reduce the dimensions of size
                        size = size[0]
        except:
            raise ProcessingMechanismError("Failed to convert size (of type {}) to a 1D array.".format(type(size)))

        if size is not None:
            size = np.array(list(map(checkAndCastInt, size)))  # convert all elements of size to int
        # except:
        #     raise ProcessingMechanismError("Failed to convert an element in size to an integer. (This "
        #                                    "should have been caught in _validate_params rather than in __init__")
        # endregion

        # region If variable is None, make it a 2D array of zeros each with length=size[i]
        # implementation note: for good coding practices, perhaps add setting to enable
        # easy change of variable's default value, which is an array of zeros at the moment
        if variable is None and size is not None:
            try:
                variable = []
                for s in size:
                    variable.append(np.zeros(s))
                variable = np.array(variable)
            except:
                raise ProcessingMechanismError("variable was not specified, but PsyNeuLink was unable to "
                                    "infer variable from the size argument, {}. size should be"
                                    " an integer or an array or list of integers. Either size or "
                                    "variable must be specified.".format(size))
        # endregion

        # region If size is None, then make it a 1D array of scalars with size[i] = length(variable[i])
        if size is None and variable is not None:
            size = []
            try:
                for input_vector in variable:
                    size.append(len(input_vector))
                size = np.array(size)
            except:
                raise ProcessingMechanismError("size was not specified, but PsyNeuLink was unable to infer size from "
                                    "the variable argument, {}. variable can be an array,"
                                    " list, a 2D array, a list of arrays, array of lists, etc. Either size or"
                                    " variable must be specified.".format(variable))
        # endregion

        # region If length(size) = 1 and variable is not None, then expand size to length(variable)
        if size is not None and variable is not None:
            if len(size) == 1 and len(variable) > 1:
                new_size = np.empty(len(variable))
                new_size.fill(size[0])
                size = new_size
        # endregion

        # IMPLEMENTATION NOTE: if variable and size are both specified as arguments, they should/will be checked
        # against each other in Component.py, during _instantiate_defaults().
        # endregion

        self.variableClassDefault = variable  # should this line be here? Ask Kristin

        params = self._assign_args_to_param_dicts(params=params,
                                                  size=size)

        super().__init__(variable=variable,
                         input_states=input_states,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _validate_inputs(self, inputs=None):
        # Let mechanism itself do validation of the input
        pass

    def _validate_params(self, request_set, target_set=None, context=None):
        """ Validate SIZE param """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Validate SIZE
        if SIZE in target_set and target_set[SIZE] is not None:
            mech_size = target_set[SIZE].copy()

            def checkAndCastInt(x):
                if not isinstance(x, numbers.Number):
                    raise ProcessingMechanismError("An element ({}) in size ({}) is not a number.".
                                                   format(x, target_set[SIZE]))
                if x < 1:
                    raise ProcessingMechanismError("An element ({}) in size ({}) is not a positive number.".
                                                   format(x, target_set[SIZE]))
                if type(x) == float and not x.is_integer():
                    raise ProcessingMechanismError("An element ({}) in size is a non-integer float.".format(x))
                try:
                    y = int(x)
                except:
                    raise ProcessingMechanismError(
                        "Failed to convert an element ({}) in size argument ({}) to an integer. size "
                        "should be a number, or iterable of numbers, which are integers or "
                        "can be converted to integers.".format(x, target_set[SIZE]))
                return int(x)
            try:
                if mech_size is not None:
                    mech_size = np.atleast_1d(mech_size)
                    if len(np.shape(mech_size)) > 1:  # number of dimensions of size > 1
                        raise ProcessingMechanismError("size ({}) should either be a number,"
                                                       " 1D array, or list".format(target_set[SIZE]))
            except:
                raise ProcessingMechanismError("Failed to convert size input ({})"
                                               " to a 1D array.".format(target_set[SIZE]))

            # try:
            if mech_size is not None:
                # convert all elements of mech_size to int, check that they are valid values (e.g. positive)
                list(map(checkAndCastInt, mech_size))
            # except:
            #     raise ProcessingMechanismError("Failed to convert an element in size argument ({}) to an integer. size "
            #                                    "should be a number, or iterable of numbers, which are integers or "
            #                                    "can be converted to integers.".format(target_set[SIZE]))

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)
