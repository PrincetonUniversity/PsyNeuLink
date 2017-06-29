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
also be used by an `AdaptiveMechanism` to modify the parameters of other components (or its own).
ProcessingMechanisms are always executed before all AdpativeMechanisms in the process and/or system to which they
belong, so that any modificatons made by the AdpativeMechanism are available to all ProcessingMechanisms in the next
round of execution.

.. _ProcessingMechanism_Creation:

Creating a ProcessingMechanism
------------------------------

A ProcessingMechanism can be created by using the standard Python method of calling the constructor for the desired
type. Some types of ProcessingMechanism (for example, `ObjectiveMechanisms <ObjectiveMechanism>`) are also created
when a system or process is created, if `learning <LINK>` and/or `control <LINK>` have been specified for it.

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
        :param size: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        """

        self.system = None

        # region Fill in and infer variable and size if they aren't specified in args
        # if variable is None and size is None:
        #     variable = self.variableClassDefault
        # if variable is None and size is None:
        #     size = [1]
            # 6/28/17: the above line of code is _very_ questionable but I think it works for now because
            # all the self.variableClassDefault values are a single zero (though the format varies widely from
            # scalars to 1D arrays to 2D arrays. This line of code essentially makes variable = [[0]] if
            # self.variableClassDefault is None.
            # This would be easily fixable (simply convert variable to 2D array, then build size based on that)
            # except, sometimes self.variableClassDefault is None! I think self.variableClassDefault
            # should simply never be None, or some other fix should be made.

        # 6/23/17: This conversion is safe but likely redundant. If, at some point in development, size and
        # variable are no longer 1D or 2D arrays, this conversion MIGHT still be safe (e.g. if they become 3D arrays).
        # region Convert variable (if given) to a 2D array, and size (if given) to a 1D integer array
        try:
            if variable is not None:
                variable = np.atleast_2d(variable)
                if len(np.shape(variable)) > 2:  # number of dimensions of variable > 2
                    warnings.warn("variable had more than two dimensions (had {} dimensions) "
                                  "so only the first element of its second-highest-numbered axis will be"
                                  " used".format(len(np.shape(variable))))
                    while len(np.shape(variable)) > 2:  # reduce the dimensions of variable
                        variable = variable[0]
        except:
            raise ProcessingMechanismError("Failed to convert variable (of type {})"
                                           " to a 2D array".format(type(variable)))

        try:
            if size is not None:
                size = np.atleast_1d(size)
                if len(np.shape(size)) > 1:  # number of dimensions of size > 1
                    warnings.warn("size had more than one dimension (size had {} dimensions), so only the first "
                                  "element of its highest-numbered axis will be used".format(len(np.shape(size))))
                    while len(np.shape(size)) > 1:  # reduce the dimensions of size
                        size = size[0]
        except:
            raise ProcessingMechanismError("Failed to convert size (of type {}) to a 1D array.".format(type(size)))

        try:
            if size is not None:
                size = np.array(list(map(lambda x: int(x), size)))  # convert all elements of size to int
        except:
            raise ProcessingMechanismError("Failed to convert an element in size to an integer. (This "
                                           "should have been caught in _validate_params rather than in __init__")
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
        if SIZE in target_set:
            mech_size = target_set[SIZE].copy()
            try:
                if mech_size is not None:
                    mech_size = np.atleast_1d(mech_size)
                    if len(np.shape(mech_size)) > 1:  # number of dimensions of size > 1
                        raise ProcessingMechanismError("size ({}) should either be a number,"
                                                       " 1D array, or list".format(mech_size))
            except:
                raise ProcessingMechanismError("Failed to convert size input (of type {})"
                                               " to a 1D array.".format(type(mech_size)))

            try:
                if mech_size is not None:
                    mech_size = list(map(lambda x: int(x), mech_size))  # convert all elements of mech_size to int
            except:
                raise ProcessingMechanismError("Failed to convert an element in size to an integer. size should "
                                               "be a number, or iterable of numbers, which are integers or"
                                               " can be converted to integers.")