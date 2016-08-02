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

from PsyNeuLink.Functions.Projections.Projection import *
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.States.ParameterState import ParameterState
from PsyNeuLink.Functions.States.OutputState import OutputState
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms import MonitoringMechanism
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms import ProcessingMechanism

# from Functions.Utility import *

# Params:
kwLearningRate = "LearningRate"
kwWeightMatrix = "Weight Matrix"
kwWeightMatrixParams = "Weight Matrix Params"


class LearningSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningSignal(Projection_Base):
    """Implement projection conveying values from output of a mechanism to input of another (default: IdentityMapping)

    Description:
        The LearningSignal class is a functionType in the Projection category of Function,
        It's execute method uses either the OutputState.value of a MonitoringMechanism or
            the receiverError attribute of a Mapping.executeMethodParameterState.receiverError
            to adjust the kwMatrix parameter (in kwExecuteMethodParams) of a receiver Mapping Projection

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
        # + defaultSender (State)
        # + defaultReceiver (State)
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
        + receiverWeightMatrix (2D np.array) - points to <Mapping>.paramsCurrent[kwExecuteMethodParams][kwMatrix]
        + weightChangeMatrix (2D np.array) - rows:  sender deltas;  columns:  receiver deltas
        + errorSignal (1D np.array) - sum of errors for each sender element of Mapping projection
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
    paramClassDefaults.update({kwProjectionSender: MonitoringMechanism, # ?? Assigned to class ref in __init__ module
                               kwExecuteMethod:BackPropagation,
                               kwExecuteMethodParams: {kwLearningRate: 1},
                               kwWeightMatrixParams: {
                                   kwExecuteMethod: LinearCombination,
                                   kwExecuteMethodParams: {kwOperation: LinearCombination.Operation.SUM},
                                   kwParamModulationOperation: ModulationOperation.ADD,
                                   # FIX: IS THIS FOLLOWING CORRECT: (WAS kwControlSignal FOR ParameterState)
                                   kwProjectionType: kwLearningSignal}
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
        """Insure sender is a MonitoringMechanism or ProcessingMechanism and receiver is a ParameterState or Mapping

        Validate send in params[kwProjectionSender] or, if not specified, sender arg:
        - must be either the outputState of a MonitoringMechanism or the errorSignal of a ProcessingMechanism, and
        - must be a list or 1D np.array (i.e., the format of an errorSignal format)

        Validate receiver in params[kwParameterStates] or, if not specified, receiver arg:
        - must be either a Mapping projection or executeMethodParameterStates[kwWeightMatrix]

         """

        # VALIDATE SENDER

        # Parse params[kwProjectionSender] if specified, and assign self.sender
        super().validate_params(request_set, target_set, context)

        # Make sure self.sender is a MonitoringMechanism or ProcessingMechanism or the outputState for one;
        # Otherwise, it should be MonitoringMechanism (assigned in paramsClassDefaults)

        sender = self.sender

        # If specified as a MonitoringMechanism, reassign to its outputState
        if isinstance(sender, MonitoringMechanism):
            self.sender = sender.outputState

        # If it is the outputState of a MonitoringMechanism, check that it is a list or 1D np.array
        if isinstance(sender, OutputState):
            if not isinstance(sender.value, (list, np.array)):
                raise LearningSignalError("Sender for {} (outputState of MonitoringMechanism {}) "
                                          "must be a list or 1D np.array".format(self.name, sender))
            if not np.array.ndim == 1:
                raise LearningSignalError("OutputState of MonitoringMechanism ({}) for {} must be an 1D np.array".
                                          format(sender, self.name))
        # If it is a ProcessingMechanism, pass (errorSignal will be assigined in instantiate_sender)
        elif isinstance(sender, ProcessingMechanism):
            pass
        # set to this as default in paramClassDefaults
        elif issubclass(sender,  MonitoringMechanism):
            pass
        else:
            raise LearningSignalError("Sender arg (or {} param ({}) for must be a MonitoringMechanism, its outputState,"
                                      " or a ProcessingMechanism".format(kwProjectionSender, sender, self.name, ))

        # VALIDATE RECEIVER
        try:
            receiver = target_set[kwParameterStates]
            self.validate_receiver(receiver)
        except (KeyError, LearningSignalError):
            # kwParameterStates not specified:
            receiver = self.receiver
            self.validate_receiver(receiver)

    def validate_receiver(self, receiver):
        # Must be a Mapping projection or the parameterState of one
        if not isinstance(receiver, (Mapping, ParameterState)):
            raise LearningSignalError("Receiver arg ({}) for {} must be a Mapping projection or a parameterState of one"
                                      .format(receiver, self.name))
        # If it is a parameterState, make sure it is the kwWeightMatrix parameter state of a Mapping projection
        if isinstance(receiver, ParameterState):
            if not receiver is receiver.owner.executeMethodParameterStates[kwWeightMatrix]:
                raise LearningSignalError("Receiver arg ({}) for {} must be the {} executeMethodParameterState of a"
                                          "Mapping projection".format(receiver, self.name, kwWeightMatrix, ))
        # Notes:
        # * if specified as a Mapping projection, it will be assigned to a parameter state in instantiate_receiver
        # * the value of receiver will be validated in instantiate_receiver

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Override super to call instantiate_receiver before calling instantiate_sender
        """
        super().instantiate_receiver(context)
        super().instantiate_sender(context=context)

    def instantiate_attributes_after_execute_method(self, context=NotImplemented):
        """Override super since it calls instantiate_receiver which has already been called above
        """
        pass

    def instantiate_sender(self, context=NotImplemented):
        """Assign self.variable to MonitoringMechanism output or self.receiver.receiverErrorSignals 
        
        Call this after instantiate_receiver, as that is needed to determine the sender (i.e., source of errorSignal)
        
        If sender arg or kwProjectionSender was specified, it has been assigned to self.sender
            and has been validated as a MonitoringMechanism, so:
            - validate that the length of its outputState.value is the same as the width (# columns) of kwMatrix 
            - assign its outputState.value as self.variable
        If sender was not specified (i.e., passed as MonitoringMechanism_Base specified in paramClassDefaults):
           if the owner of the Mapping projection projects to a MonitoringMechanism, then
               - validate that the length of its outputState.value is the same as the width (# columns) of kwMatrix 
               - assign its outputState.value as self.variable
           otherwise, if self.receiver.owner.receiver.owner has an errorSignal attribute, use that as self.variable
               (e.g., "hidden units in a multilayered neural network, using BackPropagation Function)
           [TBI: otherwise, implement default MonitoringMechanism]
           otherwise, raise exception
         
        """

# FIX: VALIDATE THAT self.sender.outputState.value IS COMPATIBLE WITH self.receiver.kwMATRIX ??LEN OR nDim or shape[0]????

        if isinstance(self.sender, MonitoringMechanism):
            # - validate that the length of its outputState.value is the same as the width (# columns) of kwMatrix 
            # - assign its outputState.value as self.variable
        else:
            if 
           # IMPLEMENT: CHECK self.receiver.owner.outputStates FOR PROJECTION TO MONITORING MECHANISM AND USE IF FOUND
           #     - validate that the length of its outputState.value is the same as the width (# columns) of kwMatrix 
           #     - assign its outputState.value as self.variable
            elif:
           # IMPLEMENT: ASSIGN??/CHECK FOR?? self.receiver.owner.receiver.owner.errorSignal AND ASSIGN TO self.variable
            else:
           # IMPLEMENT: RAISE EXCEPTION FOR MISSING MONITORING MECHANISM / SOURCE OF ERROR SIGNAL FOR LEARNING SIGNAL
           #            OR INSTANTIATE DEFAULT MONITORING MECHANISM                     

        # FIX: ??CALL:
        # super().instantiate_sender(context=context)
        


    def instantiate_receiver(self, context=NotImplemented):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a Mapping Projection, it is assigned to executeMethodParameterStates[kwWeightMatrix]
            for the projection;  if that does not exist, it is instantiated and assigned as the receiver
        If specified as a ParameterState, validate that it is executeMethodParameterStates[kwWeightMatrix]
        Validate that the LearningSignal's error matrix is the same shape as the recevier's weight matrix
        
        Notes:

        # FIX:  ??STILL TRUE: ----------
        * Requires that owner.paramsCurrent[state_param_identifier] be specified and
            set to None or to a list of state_type States
        ---------------------------------

        * This must be called before instantiate_sender since that requires access to self.receiver
            to determine whether to use a comparator mechanism or <Mapping>.receiverError for error signals
        * Doesn't call super().instantiate_receiver since that assumes self.receiver.owner is a Mechanism
                              and calls add_projection_to_mechanism
        """
# FIX: REINSTATE CALL TO SUPER AFTER GENERALIZING IT TO USE Projection.add_to

        # Validate that self.receiver is a ParameterState or a Mapping Projection

        # If receiver is a ParameterState, make sure it is executeMethodParameters[kwMatrx] parameterState
        if isinstance(self.receiver, ParameterState):
            if not self.receiver is self.receiver.owner.executeMethodParameterStates[kwWeightMatrix]:
                raise LearningSignalError("Receiver arg ({}) for {} must be the "
                                          "executeMethodParameterStates[kwWeightMatrix] of the receiver".
                                          format(self.receiver, self.name))

        # If it is not a ParameterState, it must be Mapping Projection;  else, raise exception
        elif not isinstance(self.receiver, Mapping):
            raise LearningSignalError("Receiver arg ({}) for {} must be a Mapping projection or"
                                      " a MechanismParatemerState of one".format(self.receiver, self.name))

        receiver_parameter_state_name = kwWeightMatrix

        from PsyNeuLink.Functions.States.InputState import instantiate_mechanism_state_list
        from PsyNeuLink.Functions.States.InputState import instantiate_mechanism_state
        # from Functions.States.ParameterState import ParameterState
        # from Functions.Projections.Mapping import Mapping

        # If receiver was specified as a ParameterState
        if isinstance(self.receiver, ParameterState):
            # Get owner's weight matrix (receiver.paramsCurrent[executeMethodParams][kwMatrix])
            # Note: this is a sanity check, as Mapping Projection should always have kwMatrix in paramClassDefaults
            try:
                self.receiverWeightMatrix = self.receiver.owner.paramsCurrent[kwExecuteMethodParams][kwMatrix]
            except KeyError:
                raise LearningSignal("PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".
                                     format(self.receiver.name, kwExecuteMethodParams, kwMatrix))

        # If receiver was specified as a Mapping Projection
        elif isinstance(self.receiver, Mapping):

            # Get weight matrix (receiver.paramsCurrent[executeMethodParams][kwMatrix])
            # Note: this is a sanity check, as Mapping Projection should always have kwMatrix in paramClassDefaults
            try:
                self.receiverWeightMatrix = self.receiver.paramsCurrent[kwExecuteMethodParams][kwMatrix],
            except KeyError:
                raise LearningSignal("PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".
                                     format(self.receiver.name, kwExecuteMethodParams, kwMatrix))

            weight_matrix_params = self.paramsCurrent[kwWeightMatrixParams]

            # Check if Mapping Projection has executeMethodParameterStates Ordered Dict and kwWeightMatrix entry
            try:
                self.receiver.executeMethodParameterStates[kwWeightMatrix]
            # receiver does NOT have executeMethodParameterStates attrib
            except AttributeError:
                # Instantiate executeMethodParameterStates Ordered dict
                #     with ParameterState for receiver's executeMethodParams[kwMatrix] param
                self.receiver.executeMethodParameterStates = instantiate_mechanism_state_list(
                                                                    owner=self.receiver,
                                                                    state_list=[(receiver_parameter_state_name,
                                                                                 weight_matrix_params)],
                                                                    state_type=ParameterState,
                                                                    state_param_identifier=kwParameterState,
                                                                    constraint_values=self.receiverWeightMatrix,
                                                                    constraint_values_name=kwLearningSignal,
                                                                    context=context)
                self.receiver = self.receiver.executeMethodParameterStates[receiver_parameter_state_name]

            # receiver has executeMethodParameterStates but not (yet!) one for kwWeightMatrix, so instantiate it
            except KeyError:
                # Instantiate ParameterState for kwMatrix
                self.receiver.executeMethodParameterStates[receiver_parameter_state_name] = \
                                                                    instantiate_mechanism_state(owner=self.receiver,
                                                                            state_type=ParameterState,
                                                                            state_name=receiver_parameter_state_name,
                                                                            state_spec=kwParameterState,
                                                                            state_params=weight_matrix_params,
                                                                            constraint_values=self.receiverWeightMatrix,
                                                                            constraint_values_name=kwLearningSignal,
                                                                            context=context)

            # Assign self.receiver to parameterState to be used for weight matrix param
            self.receiver = self.receiver.executeMethodParameterStates[receiver_parameter_state_name]
        # FIX: MAKE SURE MAPPING.update() USES ParameterState TO UPDATE ITS executeMethodParams
        # FIX:                                                                         (LIKE MECHANISMS DO)

        # Insure that LearningSignal output and receiver's weight matrix are same shape
        try:
            receiver_weight_matrix_shape = self.receiverWeightMatrix.shape
        except TypeError:
            # self.receiverWeightMatrix = 1
            receiver_weight_matrix_shape = 1
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

        # Add LearningSignal projection to receiver's parameterState
        self.add_to(receiver=self.receiver, state=ParameterState, context=context)


    def update(self, params=NotImplemented, context=NotImplemented):
        """

        DOCUMENT:
        LearnningSignal (Projection):
            - sender:  output of Monitoring Mechanism
                default: receiver.ownerMechanism.outputState.sendsToProjections.<MonitoringMechanism> if specified,
                         else default Comparator
            - receiver: Mapping Projection parameterState (or some equivalent thereof)

        Mapping Projection should have kwLearningParam which:
           - specifies LearningSignal
           - defaults to BP
        ?? - uses self.outputStates.sendsToProjections.<MonitoringMechanism> if specified

        LearningSignal update method:
            Generalized delta rule:
            weight = weight + (learningRate * errorDerivative * transferDerivative * sampleSender)
            for sumSquared error function:  errorDerivative = (target - sample)
            for logistic activation function: transferDerivative = sample * (1-sample)
        NEEDS:
        - errorDerivative:  get from kwExecuteMethod of Comparator Mechanism
        - transferDerivative:  get from kwExecuteMethod of Processing Mechanism

        :return: (2D np.array) self.weightChangeMatrix
        """

        # ASSIGN INPUT:
        # Array of input values from Mapping projection's sender mechanism's outputState
        # LearningSignal(self).ParameterState(receiver).Mapping(owner).OutputState(sender)
        input = self.receiver.owner.sender.value

        # ASSIGN OUTPUT
        # Array of output values for Mapping projection's recvr mech
        # LearningSignal(self).ParameterState(receiver).Mapping(owner).OutputState(receiver).ProcessMechanism(owner)
        output = self.receiver.owner.receiver.owner.value

        # ASSIGN ERROR
        # If the LearningSignal sender is a MonitoringMechanism, then the errorSignal is the just sender's value
        if isinstance(self.sender, MonitoringMechanism):
            self.errorSignal = self.sender.value
        # If the LearningSignal sender is a ProcessingMechanism, the errorSignal is sum of the contributions that a
        #    sender makes to the error of each of the receivers to which it projects, scaled by its projection weights
        elif isinstance(self.sender, ProcessingMechanism):
            self.errorSignal = np.dot(self.receiverWeightMatrix, self.sender.errorSignal)
        else:
            raise LearningSignalError("PROGRAM ERROR: unsupported Mechanism type ({})"
                                      " passed to LearningSignal {}.update()".
                                      format(self.sender.__class__.__name__, self.name))

        # CALL EXECUTE METHOD TO GET WEIGHT CHANGES
        # rows:  sender errors;  columns:  receiver errors
        self.weightChangeMatrix = self.execute([input, output, self.errorSignal], params=params, context=context)

        # Sum rows of weightChangeMatrix to get errors for each item of Mapping projection's sender
        self.weightChanges = np.add.reduce(self.weightChangeMatrix,1)

        return self.weightChangeMatrix
