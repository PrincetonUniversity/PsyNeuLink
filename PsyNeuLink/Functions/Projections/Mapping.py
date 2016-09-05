# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **********************************************  Mapping **************************************************************
#

from PsyNeuLink.Functions.Projections.Projection import *
from PsyNeuLink.Functions.Utility import *


class Mapping(Projection_Base):
    """Implement projection conveying values from output of a mechanism to input of another (default: IdentityMapping)

    Description:
        The Mapping class is a functionType in the Projection category of Function,
        It's function conveys (and possibly transforms) the OutputState.value of a sender
            to the InputState.value of a receiver

    Instantiation:
        - Mapping Projections can be instantiated in one of several ways:
            - directly: requires explicit specification of the sender
            - as part of the instantiation of a mechanism:
                the mechanism outputState will automatically be used as the receiver:
                    if the mechanism is being instantiated on its own, the sender must be explicity specified
                    if the mechanism is being instantiated within a configuration:
                        if a sender is explicitly specified for the mapping, that will be used;
                        otherwise, if it is the first mechanism in the list, process.input will be used as the sender;
                        otherwise, the preceding mechanism in the list will be used as the sender

    Initialization arguments:
        - sender (State) - source of projection input (default: systemDefaultSender)
        - receiver: (State or Mechanism) - destination of projection output (default: systemDefaultReceiver)
            if it is a Mechanism, and has >1 inputStates, projection will be mapped to the first inputState
# IMPLEMENTATION NOTE:  ABOVE WILL CHANGE IF SENDER IS ALLOWED TO BE A MECHANISM (SEE FIX ABOVE)
        - params (dict) - dictionary of projection params:
# IMPLEMENTTION NOTE: ISN'T kwProjectionSenderValue REDUNDANT WITH sender and receiver??
            + kwProjectionSenderValue (list): (default: [1]) ?? OVERRIDES sender ARG??
            + kwFunction (Utility): (default: LinearMatrix)
            + kwFunctionParams (dict): (default: {kwMatrix: kwIdentityMatrix})
# IMPLEMENTATION NOTE:  ?? IS THIS STILL CORRECT?  IF NOT, SEARCH FOR AND CORRECT IN OTHER CLASSES
        - name (str) - if it is not specified, a default based on the class is assigned in register_category
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
# IMPLEMENTATION NOTE:  AUGMENT SO THAT SENDER CAN BE A Mechanism WITH MULTIPLE OUTPUT STATES, IN WHICH CASE:
#                RECEIVER MUST EITHER BE A MECHANISM WITH SAME NUMBER OF INPUT STATES AS SENDER HAS OUTPUTSTATES
#                (FOR WHICH SENDER OUTPUTSTATE IS MAPPED TO THE CORRESONDING RECEIVER INPUT STATE
#                              USING THE SAME MAPPING PROJECTION MATRIX, OR AN ARRAY OF THEM)
#                OR BOTH MUST BE 1D ARRAYS (I.E., SINGLE VECTOR)
#       SHOULD BE CHECKED IN OVERRIDE OF validate_variable THEN HANDLED IN instantiate_sender and instantiate_receiver


    Parameters:
        The default for kwFunction is LinearMatrix using kwIdentityMatrix:
            the sender state is passed unchanged to the receiver's state
# IMPLEMENTATION NOTE:  *** CONFIRM THAT THIS IS TRUE:
        kwFunction can be set to another function, so long as it has type kwMappingFunction
        The parameters of kwFunction can be set:
            - by including them at initialization (param[kwFunction] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwFunction].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)



    ProjectionRegistry:
        All Mapping projections are registered in ProjectionRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Mapping projections can be named explicitly (using the name='<name>' argument).  If this argument is omitted,
        it will be assigned "Mapping" with a hyphenated, indexed suffix ('Mapping-n')

    Class attributes:
        + className = kwMapping
        + functionType = kwProjection
        # + defaultSender (State)
        # + defaultReceiver (State)
        + paramClassDefaults (dict)
            paramClassDefaults.update({
                               kwFunction:LinearMatrix,
                               kwFunctionParams: {
                                   # LinearMatrix.kwReceiver: receiver.value,
                                   LinearMatrix.kwMatrix: LinearMatrix.kwDefaultMatrix},
                               kwProjectionSender: kwInputState, # Assigned to class ref in __init__ module
                               kwProjectionSenderValue: [1],
                               })
        + paramNames (dict)
        # + senderDefault (State) - set to Process inputState
        + classPreference (PreferenceSet): MappingPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

    Class methods:
        function (executes function specified in params[kwFunction]

    Instance attributes:
        + sender (State)
        + receiver (State)
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramsCurrent (dict) - set currently in effect
        + variable (value) - used as input to projection's function
        + value (value) - output of function
        + monitoringMechanism (MonitoringMechanism) - source of error signal for matrix weight changes
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying MappingPreferenceSet

    Instance methods:
        none
    """

    functionType = kwMapping
    className = functionType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwFunction: LinearMatrix,
                               kwProjectionSender: kwOutputState, # Assigned to class ref in __init__.py module
                               kwProjectionSenderValue: [1],
                               })

    def __init__(self,
                 sender=NotImplemented,
                 receiver=NotImplemented,
                 # function=LinearMatrix(matrix=kwDefaultMatrix),
                 matrix=kwDefaultMatrix,
                 param_modulation_operation=ModulationOperation.ADD,
                 params=None,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """
IMPLEMENTATION NOTE:  *** DOCUMENTATION NEEDED (SEE CONTROL SIGNAL)

        :param sender:
        :param receiver:
        :param params:
        :param name:
        :param context:
        :return:
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(
                                                 # function=function,
                                                 function_params={kwMatrix: matrix},
                                                 param_modulation_operation=param_modulation_operation,
                                                 params=params)

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        self.monitoringMechanism = None

        # MODIFIED 9/2/16 ADDED:
        # If receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        if receiver is NotImplemented:
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = self
            self.init_args['name'] = name

            # Flag for deferred initialization
            self.value = kwDeferredInit
            return
        # MODIFIED 9/2/16 END

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super(Mapping, self).__init__(sender=sender,
                                      receiver=receiver,
                                      params=params,
                                      name=name,
                                      prefs=prefs,
                                      context=self)

    def instantiate_receiver(self, context=NotImplemented):
        """Handle situation in which self.receiver was specified as a Mechanism (rather than State)

        If receiver is specified as a Mechanism, it is reassigned to the (primary) inputState for that Mechanism
        If the Mechanism has more than one inputState, assignment to other inputStates must be done explicitly
            (i.e., by: instantiate_receiver(State)

        """
        # Assume that if receiver was specified as a Mechanism, it should be assigned to its (primary) inputState
        if isinstance(self.receiver, Mechanism):
            if (len(self.receiver.inputStates) > 1 and
                    (self.prefs.verbosePref or self.receiver.prefs.verbosePref)):
                print("{0} has more than one inputState; {1} was assigned to the first one".
                      format(self.receiver.owner.name, self.name))
            self.receiver = self.receiver.inputState

        # Compare length of Mapping output and receiver's variable to be sure matrix has proper dimensions
        try:
            receiver_len = len(self.receiver.variable)
        except TypeError:
            receiver_len = 1
        try:
            mapping_input_len = len(self.value)
        except TypeError:
            mapping_input_len = 1

        if receiver_len != mapping_input_len:
            from PsyNeuLink.Functions.States.ParameterState import get_function_param
            matrix_spec = get_function_param(self.paramsCurrent[kwFunctionParams][kwMatrix])

            # IMPLEMENT: INCLUDE OPTION TO ALLOW RECONFIGURATION
            self.reshapeWeightMatrixOption = True
            if self.reshapeWeightMatrixOption and (matrix_spec is kwFullConnectivityMatrix or
                             matrix_spec is kwIdentityMatrix):
                    # self.matrix = np.full((len(self.variable), receiver_len),1.0)
                    self.matrix = np.random.rand(len(self.variable), receiver_len)
            # if it is a function, assume it uses random.rand() and call with sender and receiver lengths
            elif self.reshapeWeightMatrixOption and isinstance(matrix_spec, function_type):
                    self.matrix = matrix_spec(len(self.variable), receiver_len)
            else:
                raise ProjectionError("Length ({0}) of output for {1} projection from {2}"
                                      " must equal length ({3}) of {4} inputState".
                                      format(mapping_input_len,
                                             self.name,
                                             self.sender.name,
                                             receiver_len,
                                             self.receiver.owner.name))

        super(Mapping, self).instantiate_receiver(context=context)

    def execute(self, input=NotImplemented, params=NotImplemented, time_scale=NotImplemented, context=NotImplemented):
        # IMPLEMENT: check for flag that it has changed (needs to be implemented, and set by ErrorMonitoringMechanism)
        # DOCUMENT: update, including use of monitoringMechanism.monitoredStateChanged and weightChanged flag
        """
        If there is an functionParrameterStates[kwLearningSignal], update the matrix parameterState:
                 it should set params[kwParameterStateParams] = {kwLinearCombinationOperation:SUM (OR ADD??)}
                 and then call its super().execute
           - use its value to update kwMatrix using CombinationOperation (see State update ??execute method??)

        """

        # ASSUMES IF self.monitoringMechanism IS ASSIGNED AND parameterState[kwMatrix] HAS BEEN INSTANTIATED
        # AVERTS DUCK TYPING WHICH OTHERWISE WOULD BE REQUIRED FOR THE MOST FREQUENT CASES (I.E., NO LearningSignal)

        # Check whether weights changed
        if self.monitoringMechanism and self.monitoringMechanism.summedErrorSignal:

            # Assume that if monitoringMechanism attribute is assigned,
            #    both a LearningSignal and parameterState[kwMatrix] to receive it have been instantiated
            matrix_parameter_state = self.parameterStates[kwMatrix]

            # Assign current kwMatrix to parameter state's baseValue, so that it is updated in call to execute()
            matrix_parameter_state.baseValue = self.matrix

            # Pass params for parameterState's funtion specified by instantiation in LearningSignal
            weight_change_params = matrix_parameter_state.paramsCurrent

            # Update parameter state: combines weightChangeMatrix from LearningSignal with matrix baseValue
            matrix_parameter_state.update(weight_change_params, context=context)

            # Update kwMatrix
            self.matrix = matrix_parameter_state.value

        return self.function(self.sender.value, params=params, context=context)

    @property
    def matrix(self):
        return self.function.__self__.matrix

    @matrix.setter
    def matrix(self, matrix):
        # FIX: ADD VALIDATION OF MATRIX AND/OR 2D np.array HERE??
        self.function.__self__.matrix = matrix
