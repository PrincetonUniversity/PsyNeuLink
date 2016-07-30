# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************  LearningSignal **********************************************************
#

from Functions.Projections.Projection import *
from Functions.Projections.Mapping import Mapping
from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
# from Functions.Utility import *

# Params:
kwLearningRate = "LearningRate"

class LearningSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningSignal(Projection_Base):
    """Implement projection conveying values from output of a mechanism to input of another (default: IdentityMapping)

    Description:
        The LearningSignal class is a functionType in the Projection category of Function,
        It's execute method uses the MechanismOutputState.value of a MonitoringMechanism
            to adjust the kwMatrix parameter of (in kwExecuteMethodParams) of a receiver Mapping Projection

    Instantiation:
        - LearningSignal Projections are instantiated by specifying a MonitoringMechanism sender and a Mapping receiver

    Initialization arguments:
        - sender (MonitoringMechanism) - source of projection input (default: TBI)
        - receiver: (Mapping Projection) - destination of projection output (default: TBI)
        - params (dict) - dictionary of projection params:
            + kwExecuteMethod (Utility): (default: BP)
            + kwExecuteMethodParams (dict):
                + kwLearningRate (value): (default: 1)
        - name (str) - if it is not specified, a default based on the class is assigned in register_category
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)

    Parameters:
        The default for kwExecuteMethod is BackPropagation:
        The parameters of kwExecuteMethod can be set:
            - by including them at initialization (param[kwExecuteMethod] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwExecuteMethod].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)

    ProjectionRegistry:
        All LearningSignal projections are registered in ProjectionRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        LearningSignal projections can be named explicitly (using the name argument).  If this argument is omitted,
        it will be assigned "LearningSignal" with a hyphenated, indexed suffix ('LearningSignal-n')

    Class attributes:
        + className = kwLearningSignal
        + functionType = kwProjection
        # + defaultSender (MechanismState)
        # + defaultReceiver (MechanismState)
        + paramClassDefaults (dict):
            + kwExecuteMethod (Utility): (default: BP)
            + kwExecuteMethodParams:
                + kwLearningRate (value): (default: 1)
        + paramNames (dict)
        + classPreference (PreferenceSet): LearningSignalPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

    Class methods:
        function (executes function specified in params[kwExecuteMethod]

    Instance attributes:
        + sender (MonitoringMechanism)
        + receiver (Mapping)
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramsCurrent (dict) - set currently in effect
        + variable (value) - used as input to projection's execute method
        + value (value) - output of execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying LearningSignalPreferenceSet

    Instance methods:
        none
    """

    functionType = kwLearningSignal
    className = functionType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwExecuteMethod:kwBP,
                               kwExecuteMethodParams: {
                                   kwLearningRate: 1}
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
        super().__init__(sender=sender,
                         receiver=receiver,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Insure that sender is a MonitoringMechanism, the output of which is compatible with self.variable
         """

        #
        # try:
        #     param_value = params[kwMechanismInputStates]
        #
        # except KeyError:
        #     # kwMechanismInputStates not specified:
        #     # - set to None, so that it is set to default (self.variable) in instantiate_inputState
        #     # - if in VERBOSE mode, warn in instantiate_inputState, where default value is known
        #     params[kwMechanismInputStates] = None

        super().validate_params(request_set, target_set, context)

        # FIX: MAKE SURE self.sender / kwSender IS A MONITORING MECHANISM AND THAT

        # FIX: INSTANTIATE AND/OR ASSIGN MonitoringMechanism AND ERROR VECTOR  IS SAME LENGTH AS RELEVANT DIM OF WEIGHT MATRIX
        # FIX: MAKE SURE LENGTH OF ERROR VECTOR FROM SENDER IS SAME AS RELEVANT PARAM OF EXECUTE METHOD FOR ERROR COMPUTATION


    def instantiate_sender(self, context=NotImplemented):
        """Instantiate and assign default MonitoringMechanism if necessary

        :param context:
        :return:
        """

        # FIX:  CALL?
        # FIX: NEED TO IMPLEMENT DEFAULT COMPARATOR
        super().instantiate_sender(context=context)


    def instantiate_receiver(self, context=NotImplemented):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a Mapping Projection, it is assigned to executeMethodParameterStates[kwMatrix]
            for the projection;  if that does not exist, it is instantiated and assigned as the receiver
        If it is specified as a MechanismParameterState, validate that it is executeMethodParameterStates[kwMatrix]
        Validate that the LearningSignal's error matrix is the same shape as the recevier's weight matrix
        Call super().instantiate_receiver

        """

        # Validate that self.receiver is a MechanismParameterState or a Mapping Projection

        # If receiver is a MechanismParameterState, make sure it is executeMethodParameters[kwMatrx] parameterState
        if isinstance(self.receiver, MechanismParameterState):
            if not self.receiver is self.receiver.owner.executeMethodParameterStates[kwMatrix]:
                raise LearningSignalError("Receiver arg ({}) for {} must be the executeMethodParameterStates[kwMatrix]"
                                          " of the receiver".format(self.receiver, self.name))

        # If it is not a MechanismParameterState, it must be Mapping Projection;  else, raise exception
        elif not isinstance(self.receiver, Mapping):
            raise LearningSignalError("Receiver arg ({}) for {} must be a Mapping projection or"
                                      " a MechanismParatemerState of one".format(self.receiver, self.name))

        receiver_parameter_state_name = kwMatrix

        from Functions.MechanismStates.MechanismInputState import instantiate_mechanism_state_list
        from Functions.MechanismStates.MechanismInputState import instantiate_mechanism_state
        # from Functions.MechanismStates.MechanismParameterState import MechanismParameterState
        # from Functions.Projections.Mapping import Mapping

        # If receiver was specified as a MechanismParameterState
        if isinstance(self.receiver, MechanismParameterState):
            # Get owne'rs weight matrix (receiver.paramsCurrent[executeMethodParams][kwMatrix])
            # Note: this is a sanity check, as Mapping Projection should always have kwMatrix in paramClassDefaults
            try:
                receiver_weight_matrix = self.receiver.owner.paramsCurrent[kwExecuteMethodParams][kwMatrix]
            except KeyError:
                raise LearningSignal("PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".
                                     format(self.receiver.name, kwExecuteMethodParams, kwMatrix))

        # If receiver was specified as a Mapping Projection
        elif isinstance(self.receiver, Mapping):

            # Get weight matrix (receiver.paramsCurrent[executeMethodParams][kwMatrix])
            # Note: this is a sanity check, as Mapping Projection should always have kwMatrix in paramClassDefaults
            try:
                receiver_weight_matrix = self.receiver.paramsCurrent[kwExecuteMethodParams][kwMatrix],
            except KeyError:
                raise LearningSignal("PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".
                                     format(self.receiver.name, kwExecuteMethodParams, kwMatrix))

            # Check if Mapping Projection has executeMethodParameterStates Ordered Dict and kwMatrix entry
            try:
                self.receiver.executeMethodParameterStates[kwMatrix]
            # receiver has no executeMethodParameterStates
            except AttributeError:
                # Instantiate executeMethodParameterStates Ordered dict with MechanismParameterState for kwMatrix param
                self.receiver.executeMethodParameterStates = instantiate_mechanism_state_list(
                                                                    owner=self.receiver,
                                                                    state_list=[receiver_parameter_state_name],
                                                                    state_type=MechanismParameterState,
                                                                    state_param_identifier=kwMechanismParameterState,
                                                                    constraint_values=receiver_weight_matrix,
                                                                    constraint_values_name=kwLearningSignal,
                                                                    context=context)
                self.receiver = self.receiver.executeMethodParameterStates[kwMatrix]

            # receiver has executeMethodParameterStates but not (yet!) one for kwMatrix, so instantiate it
            except KeyError:
                # Instantiate MechanismParameterState for kwMatrix
                self.receiver.executeMethodParameterStates[receiver_parameter_state_name] = \
                                                                    instantiate_mechanism_state(owner=self.receiver,
                                                                            state_type=MechanismParameterState,
                                                                            state_name=receiver_parameter_state_name,
                                                                            state_spec=kwMechanismParameterState,
                                                                            constraint_values=receiver_weight_matrix,
                                                                            constraint_values_name=kwLearningSignal,
                                                                            context=context)

            # Assign self.receiver to parameterState to be used for weight matrix param
            self.receiver = self.receiver.executeMethodParameterStates[kwMatrix]
        # FIX: MAKE SURE MAPPING PROJECTION executeMethod USES MechanismParameterState TO UPDATE ITS executeMethodParams
        # FIX:                                                                         (LIKE MECHANISMS DO)
        # FIX: NEED TO ASSIGN AGGREGATION OPERATION TO BE ADDITIVE (RATHER THAN MULTIPLICATIVE) BY DEFAULT

        # Insure that LearningSignal output and receiver's weight matrix are same shape
        try:
            receiver_weight_matrix_shape = receiver_weight_matrix.shape
        except TypeError:
            receiver_weight_matrix = 1
        try:
            learning_signal_shape = self.value.shape
        except TypeError:
            learning_signal_shape = 1

        if receiver_weight_matrix_shape != learning_signal_shape:
            raise ProjectionError("Shape ({0}) of matrix for {1} learning signal from {2}"
                                  " must match shape of receiver weight matrix ({3}) for {4}".
                                  format(learning_signal_shape,
                                         self.name,
                                         self.sender.name,
                                         receiver_weight_matrix_shape,
                                         self.receiver.owner.name))

        # # IMPLEMENTATION NOTE:  Don't call this as it assumes self.receiver.owner is a Mechanism
        # #                       and calls add_projection_to_mechanism
        # super().instantiate_receiver(context=context)

        # Add LearningSignal projection to receiver's parameterState
        self.add_to(receiver=self.receiver, state=MechanismParameterState, context=context)


    def update(self, params=NotImplemented, context=NotImplemented):
        """
# 2) LearnningSignal (Projection):
#     - sender:  output of Monitoring Mechanism
#         default: receiver.ownerMechanism.outputState.sendsToProjections.<MonitoringMechanism> if specified,
#                  else default Comparator
#     - receiver: Mapping Projection parameterState (or some equivalent thereof)
#
# Mapping Projection should have kwLearningParam which:
#    - specifies LearningSignal
#    - defaults to BP
# ?? - uses self.outputStates.sendsToProjections.<MonitoringMechanism> if specified
#
# LearningSignal update method:
# Generalized delta rule:
# weight = weight + (learningRate * errorDerivative * transferDerivative * sampleSender)
# for sumSquared error function:  errorDerivative = (target - sample)
# for logistic activation function: transferDerivative = sample * (1-sample)
# NEEDS:
# - errorDerivative:  get from kwExecuteMethod of Comparator Mechanism
# - transferDerivative:  get from kwExecuteMethod of Process Processing Mechanism
        :return:
        """

        # IMPLEMENTATION NOTE:  ADD LearningSignal HERE IN FUTURE
        # super(Mapping, self).update(params=, context)




        weight_changes = self.sender.value
        weights_matrix = self.receiver.ownerMechanism.executeMethodParams[kwMatrix]



        return self.execute(self.sender.value, params=params, context=context)
