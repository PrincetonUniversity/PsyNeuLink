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
        It's execute method conveys (and possibly transforms) the OutputState.value of a sender
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
            + kwExecuteMethod (Utility): (default: LinearMatrix)
            + kwExecuteMethodParams (dict): (default: {kwMatrix: kwIdentityMatrix})
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
        The default for kwExecuteMethod is LinearMatrix using kwIdentityMatrix:
            the sender state is passed unchanged to the receiver's state
# IMPLEMENTATION NOTE:  *** CONFIRM THAT THIS IS TRUE:
        kwExecuteMethod can be set to another function, so long as it has type kwMappingFunction
        The parameters of kwExecuteMethod can be set:
            - by including them at initialization (param[kwExecuteMethod] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwExecuteMethod].adjust(params)
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
                               kwExecuteMethod:LinearMatrix,
                               kwExecuteMethodParams: {
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
        function (executes function specified in params[kwExecuteMethod]

    Instance attributes:
        + sender (State)
        + receiver (State)
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramsCurrent (dict) - set currently in effect
        + variable (value) - used as input to projection's execute method
        + value (value) - output of execute method
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
    paramClassDefaults.update({kwExecuteMethod:LinearMatrix,
                               kwExecuteMethodParams: {
                                   # LinearMatrix.kwReceiver: receiver.value,
                                   # FIX: ADD CAPABILITY FOR TUPLE THAT ALLOWS LearningSignal TO BE SPECIFIED
                                   # FIX: SEE Mechanism HANDLING OF ControlSignal Projection SPECIFICATION
                                   kwMatrix: kwDefaultMatrix},
                               kwProjectionSender: kwOutputState, # Assigned to class ref in __init__.py module
                               kwProjectionSenderValue: [1],
                               })

    def __init__(self,
                 sender=NotImplemented,
                 receiver=NotImplemented,
                 params=NotImplemented,
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

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super(Mapping, self).__init__(sender=sender,
                                      receiver=receiver,
                                      params=params,
                                      name=name,
                                      prefs=prefs,
                                      context=self)
        TEST = True

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):

        super().instantiate_attributes_before_execute_method(context)

        try:
            self.paramsCurrent[kwLearningSignal]
        except KeyError:
            pass
        else:
            self.instantiate_parameter_states(context=context)

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


        # Insure that Mapping output and receiver's variable are the same length
        try:
            receiver_len = len(self.receiver.variable)
        except TypeError:
            receiver_len = 1
        try:
            mapping_input_len = len(self.value)
        except TypeError:
            mapping_input_len = 1

        if receiver_len != mapping_input_len:
            raise ProjectionError("Length ({0}) of output for {1} projection from {2}"
                                  " must equal length ({3}) of {4} inputState".
                                  format(mapping_input_len,
                                         self.name,
                                         self.sender.name,
                                         receiver_len,
                                         self.receiver.owner.name))

        super(Mapping, self).instantiate_receiver(context=context)

    def update(self, params=NotImplemented, context=NotImplemented):
        # IMPLEMENT: check for flag that it has changed (needs to be implemented, and set by ErrorMonitoringMechanism)
        """
        If there is an executeMethodParrameterStates[kwLearningSignal], update it:
                 it should set params[kwParameterStateParams] = {kwLinearCombinationOperation:SUM (OR ADD??)}
                 and then call its super().update
           - use its value to update kwMatrix using CombinationOperation (see State update method)

        """

        from PsyNeuLink.Functions.Projections.LearningSignal import kwWeightChangeMatrix
        try:
            weight_change_parameter_state = self.parameterStates[kwWeightChangeMatrix]

        except:
            pass

        else:
            # Assign current kwMatrix to parameter state's baseValue, so that it is updated in call to update()
            weight_change_parameter_state.baseValue = self.paramsCurrent[kwMatrix]

            # Pass params for parameterState's execute method specified by instantiation in LearningSignal
            params = {kwParameterStateParams: weight_change_parameter_state.paramsCurrent}

            # Update parameter state, which combines weightChangeMatrix from LearningSignal with self.baseValue
            weight_change_parameter_state.update(params, context)

            # Update kwMatrix
            self.paramsCurrent[kwMatrix] = weight_change_parameter_state.value

        return self.execute(self.sender.value, params=params, context=context)
