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
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.MonitoringMechanism import MonitoringMechanism_Base
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.LinearComparator import LinearComparator
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.WeightedError import WeightedError
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms import ProcessingMechanism
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base

# from Functions.Utility import *

# Params:
kwLearningRate = "LearningRate"
kwWeightChangeParams = "Weight Change Params"
# kwMatrix = "Weight Change Matrix"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

DefaultTrainingMechanism = LinearComparator

class LearningSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningSignal(Projection_Base):
    """Implement projection conveying values from output of a mechanism to input of another (default: IdentityMapping)

    Description:
        The LearningSignal class is a functionType in the Projection category of Function,
        It's execute method takes the output of a MonitoringMechanism (self.variable), and the input and output of
            the ProcessingMechanism to which its receiver Mapping Projection projects, and generates a matrix of
            weight changes for the Mapping Projection's matrix parameter

    Instantiation:
        LearningSignal Projections are instantiated:
            - directly by specifying a MonitoringMechanism sender and a Mapping receiver
            - automatically by specifying the kwLearningSignal parameter of a Mapping Projection

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
    paramClassDefaults.update({kwProjectionSender: MonitoringMechanism_Base,
                               kwExecuteMethod:BackPropagation,
                               kwExecuteMethodParams: {kwLearningRate: 1,
                                                       kwParameterStates: None # This suppresses parameterStates
                                                       },
                               kwWeightChangeParams: {
                                   kwExecuteMethod: LinearCombination,
                                   kwExecuteMethodParams: {kwOperation: LinearCombination.Operation.SUM},
                                   kwParamModulationOperation: ModulationOperation.ADD,
                                   # FIX: IS THIS FOLLOWING CORRECT: (WAS kwControlSignal FOR ParameterState)
                                   # kwParameterStates: None, # This suppresses parameterStates
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
        - must be the outputState of a MonitoringMechanism (e.g., LinearComparator or WeightedError)
        - must be a list or 1D np.array (i.e., the format of an errorSignal)

        Validate receiver in params[kwParameterStates] or, if not specified, receiver arg:
        - must be either a Mapping projection or parameterStates[kwMatrix]

         """

        # VALIDATE SENDER

        # Parse params[kwProjectionSender] if specified, and assign self.sender
        super().validate_params(request_set, target_set, context)

        # Make sure self.sender is a MonitoringMechanism or ProcessingMechanism or the outputState for one;
        # Otherwise, it should be MonitoringMechanism (assigned in paramsClassDefaults)

        sender = self.sender

        # If specified as a MonitoringMechanism, reassign to its outputState
        if isinstance(sender, MonitoringMechanism_Base):
            self.sender = sender.outputState

        # If it is the outputState of a MonitoringMechanism, check that it is a list or 1D np.array
        if isinstance(sender, OutputState):
            if not isinstance(sender.value, (list, np.array)):
                raise LearningSignalError("Sender for {} (outputState of MonitoringMechanism {}) "
                                          "must be a list or 1D np.array".format(self.name, sender))
            if not np.array.ndim == 1:
                raise LearningSignalError("OutputState of MonitoringMechanism ({}) for {} must be an 1D np.array".
                                          format(sender, self.name))

        # IMPLEMENTATION NOTE:  No longer supported;  must be an instantiated MonitoringMechanism object
        # # If it is a ProcessingMechanism, pass (errorSignal will be assigined in instantiate_sender)
        # elif isinstance(sender, ProcessingMechanism_Base):
        #     pass

        # # If specification is a MonitoringMechanism class, pass (it will be instantiated in instantiate_sender)
        elif inspect.isclass(sender) and issubclass(sender,  MonitoringMechanism_Base):
            pass

        else:
            raise LearningSignalError("Sender arg (or {} param ({}) for must be a "
                                      "MonitoringMechanism or the outputState of one"
                                      .format(kwProjectionSender, sender, self.name, ))

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
        # If it is a parameterState and the receiver already has a parameterStates dict
        #     make sure the assignment is to its kwMatrix entry
        if isinstance(receiver, ParameterState):
            if receiver.owner.parameterStates and not receiver is receiver.owner.parameterStates[kwMatrix]:
                raise LearningSignalError("Receiver arg ({}) for {} must be the {} parameterState of a"
                                          "Mapping projection".format(receiver, self.name, kwMatrix, ))
        # Notes:
        # * if specified as a Mapping projection, it will be assigned to a parameter state in instantiate_receiver
        # * the value of receiver will be validated in instantiate_receiver

    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        """Override super to call instantiate_receiver before calling instantiate_sender

        Call instantiate_receiver first since both instantiate_sender and instantiate_execute_method
            reference the receiver's weight matrix: self.receiver.owner.params[executeMethodParams][kwMatrix]

        """
        # FIX: PROBLEM: instantiate_receiver usually follows instantiate_execute_method,
        # FIX:          and uses self.value (output of execute method) to validate against receiver.variable

        # # MODIFIED 8/7/16 OLD:
        # super().instantiate_receiver(context)
        # MODIFIED 8/7/16 NEW:
        self.instantiate_receiver(context)
        self.instantiate_sender(context=context)

    def instantiate_attributes_after_execute_method(self, context=NotImplemented):
        """Override super since it calls instantiate_receiver which has already been called above
        """
        pass

    def instantiate_receiver(self, context=NotImplemented):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a Mapping Projection, assign to parameterStates[kwMatrix]
            for the projection;  if that does not exist, instantiate and assign as the receiver
        If specified as a ParameterState, validate that it is parameterStates[kwMatrix]
        Validate that the LearningSignal's error matrix is the same shape as the recevier's weight matrix
        
        Notes:
        * This must be called before instantiate_sender since that requires access to self.receiver
            to determine whether to use a comparator mechanism or <Mapping>.receiverError for error signals
        * Doesn't call super().instantiate_receiver since that assumes self.receiver.owner is a Mechanism
                              and calls add_projection_to_mechanism
        """
# FIX: ??REINSTATE CALL TO SUPER AFTER GENERALIZING IT TO USE Projection.add_to
# FIX: OR, MAKE SURE FUNCTIONALITY IS COMPARABLE

        # VALIDATE that self.receiver is a ParameterState or a Mapping Projection

        # If receiver is a ParameterState, and receiver's parameterStates dict has been instantiated,
        #    make sure LearningSignal is being assigned to parameterStates[kwMatrix]
        if isinstance(self.receiver, ParameterState):

            if (self.receiver.owner.parameterStates and
                    not self.receiver is self.receiver.owner.parameterStates[kwMatrix]):
                raise LearningSignalError("Receiver arg ({}) for {} must be the "
                                          "parameterStates[{}] of the receiver".
                                          format(self.receiver, self.name, kwMatrix))

        # Receiver was specified as a Mapping Projection
        elif isinstance(self.receiver, Mapping):

            from PsyNeuLink.Functions.States.InputState import instantiate_state_list
            from PsyNeuLink.Functions.States.InputState import instantiate_state

            # Get weight matrix (receiver.paramsCurrent[executeMethodParams][kwMatrix])
            # Note: this is a sanity check, as Mapping Projection should always have kwMatrix in paramClassDefaults

            weight_change_params = self.paramsCurrent[kwWeightChangeParams]

            # Check if Mapping Projection has parameterStates Ordered Dict and kwMatrix entry
            try:
                self.receiver.parameterStates[kwMatrix]
            # receiver does NOT have parameterStates attrib
            except AttributeError:
                # Instantiate parameterStates Ordered dict
                #     with ParameterState for receiver's executeMethodParams[kwMatrix] param
                self.receiver.parameterStates = instantiate_state_list(owner=self.receiver,
                                                                       state_list=[(kwMatrix,
                                                                                    weight_change_params)],
                                                                       state_type=ParameterState,
                                                                       state_param_identifier=kwParameterState,
                                                                       constraint_value=self.receiverWeightMatrix,
                                                                       constraint_value_name=kwLearningSignal,
                                                                       context=context)
                self.receiver = self.receiver.parameterStates[kwMatrix]

            # receiver has parameterStates but not (yet!) one for kwMatrix, so instantiate it
            except KeyError:
                # Instantiate ParameterState for kwMatrix
                self.receiver.parameterStates[kwMatrix] = instantiate_state(owner=self.receiver,
                                                                            state_type=ParameterState,
                                                                            state_name=kwMatrix,
                                                                            state_spec=kwParameterState,
                                                                            state_params=weight_change_params,
                                                                            constraint_value=self.receiverWeightMatrix,
                                                                            constraint_value_name=kwLearningSignal,
                                                                            context=context)

            # Assign self.receiver to parameterState used for weight matrix param
            self.receiver = self.receiver.parameterStates[kwMatrix]

        # If it is not a ParameterState or Mapping Projection, raise exception
        else:
            raise LearningSignalError("Receiver arg ({}) for {} must be a Mapping projection or"
                                      " a MechanismParatemerState of one".format(self.receiver, self.name))

        # GET RECEIVER'S WEIGHT MATRIX
        self.get_receiver_weight_matrix()


        # # IMPLEMENTATION NOTE:  self.value (weight change matrix) NOT DEFINED YET, SO MOVED THIS TO instantiate_sender
        # # Insure that LearningSignal output and receiver's weight matrix are same shape
        # try:
        #     receiver_weight_matrix_shape = self.receiverWeightMatrix.shape
        # except TypeError:
        #     # self.receiverWeightMatrix = 1
        #     receiver_weight_matrix_shape = 1
        # try:
        #     learning_signal_shape = self.value.shape
        # except TypeError:
        #     learning_signal_shape = 1
        #
        # if receiver_weight_matrix_shape != learning_signal_shape:
        #     raise ProjectionError("Shape ({0}) of matrix for {1} learning signal from {2}"
        #                           " must match shape of receiver weight matrix ({3}) for {4}".
        #                           format(learning_signal_shape,
        #                                  self.name,
        #                                  self.sender.name,
        #                                  receiver_weight_matrix_shape,
        #                                  self.receiver.owner.name))

        # # IMPLEMENTATION NOTE:  self.value (weight change matrix) NOT DEFINED YET, SO MOVED THIS TO instantiate_sender
        # # Add LearningSignal projection to receiver's parameterState
        # # self.add_to(receiver=self.receiver, state=ParameterState, context=context)
        # self.add_to(receiver=self.receiver.owner, state=self.receiver, context=context)

    def get_receiver_weight_matrix(self):
        """Get weight matrix for Mapping projection to which LearningSignal projects

        Notes:
        * use receiver parameterState's variable, rather than its value or params[kwExecuteMethodParams][kwMatrix],
            since its executeMethod is LinearCombination, so it reduces 2D np.array (matrix) to 1D np.array (vector)
            and params[kwExecuteMethodParams][kwMatrix] may not yet have been parsed (e.g., may be a str or tuple)

        """
        # FIX: *** NEED TO GET SIZE OF MATRIX,
        # FIX: SINCE THIS IS CALLED BEFORE THE MAPPING PROJECTION HAS INSTANTIATED ITS EXECUTE METHOD

        message = "PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".format(self.receiver.name,
                                                                                             kwExecuteMethodParams,
                                                                                             kwMatrix)
        if isinstance(self.receiver, ParameterState):
            try:
                # self.receiverWeightMatrix = self.receiver.owner.paramsCurrent[kwExecuteMethodParams][kwMatrix]
                self.receiverWeightMatrix = self.receiver.variable
            except KeyError:
                raise LearningSignal(message)

        elif isinstance(self.receiver, Mapping):
            try:
                # self.receiverWeightMatrix = self.receiver.paramsCurrent[kwExecuteMethodParams][kwMatrix],
                # IMPLEMENTATION NOTE: use variable, since parameterState executeMethod is LinearCombination,
                #                      which reduces 2D np.array (matrix) to 1D np.array (vector)
                self.receiverWeightMatrix = self.receiver.parameterStates[kwMatrix].variable
            except KeyError:
                raise LearningSignal(message)


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

FROM TODO:
#    - instantiate_sender:
#        - examine mechanism to which Mapping projection projects:  self.receiver.owner.receiver.owner
#            - check if it is a terminal mechanism in the system:
#                - if so, assign:
#                    - LinearComparator MonitoringMechanism
#                        - ProcessInputState for LinearComparator (name it??) with projection to target inputState
#                        - Mapping projection from terminal ProcessingMechanism to LinearCompator sample inputState
#                - if not, assign:
#                    - WeightedSum MonitoringMechanism
#                        - Mapping projection from preceding MonitoringMechanism:
#                            preceding processing mechanism (ppm):
#                                ppm = self.receiver.owner.receiver.owner
#                            preceding processing mechanism's output projection (pop)
#                                pop = ppm.outputState.projections[0]
#                            preceding processing mechanism's output projection learning signal (popls):
#                                popls = pop.parameterState.receivesFromProjections[0]
#                            preceding MonitoringMechanism (pem):
#                                pem = popls.sender.owner
#                            assign Mapping projection from pem.outputState to self.inputState
#                        - Get weight matrix for pop (pwm):
#                                pwm = pop.parameterState.params[kwMatrix]


# HAS SENDER:
    # VALIDATE
# HAS NO SENDER:
    # error_source PROJECTS TO A MONITORING MECHANISM
    #         assign it as sender
    # error_source DOESN'T PROJECT TO A MONITORING MECHANISM
        # error_source PROJECTS TO A PROCESSING MECHANISM:
            # INSTANTIATE WeightedSum MonitoringMechanism
        # error_source PROJECTS DOESN'T PROJECT TO A PROCESSING MECHANISM:
            # INSTANTIATE DefaultTrainingMechanism

        """

        # FIX: 8/7/16
        # FIX: NEED TO DEAL HERE WITH CLASS SPECIFICATION OF MonitoringMechanism AS DEFAULT
        # FIX: OR HAVE ALREADY INSTANTIATED DEFAULT MONITORING MECHANISM BEFORE REACHING HERE
        # FIX: PARALLEL HOW DefaultMechanism (for Mapping) AND DefaultController (for ControlSignal) ARE HANDLED

        # MonitoringMechanism specified as sender
        if isinstance(self.sender, MonitoringMechanism_Base):
            # - validate that the length of the sender's outputState.value (the error signal)
            #     is the same as the width (# columns) of kwMatrix (# of receivers)
            # - assign its outputState.value as self.variable
            if len(self.sender.outputState.value) == len(self.receiverWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]):
                # FIX: SHOULD THIS BE self.inputValue?? or self.inputState.variable??
                self.variable = self.sender.outputState
            else:
                raise LearningSignalError("Length ({}) of MonitoringMechanism outputState specified as sender for {} "
                                          "must match the receiver dimension ({}) of the weight matrix for {}".
                                          format(len(self.sender.outputState.value),
                                                 self.name,
                                                 len(self.receiverWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]),
                                                 self.receiver.owner))

        # No MonitoringMechanism was specified, so instantiate one
        else:
            # Get error_source:  ProcessingMechanism for which error is being monitored
            #    (the mechanism to which the Mapping projection projects)
            # Note: Mapping.instantiate_receiver has not yet been called, so need to do parse below
            from PsyNeuLink.Functions.States.InputState import InputState
            if isinstance(self.receiver.owner.receiver, Mechanism):
                error_source = self.receiver.owner.receiver
            elif isinstance(self.receiver.owner.receiver, InputState):
                error_source = self.receiver.owner.receiver.owner

            monitoring_mechanism = None
            next_level_monitoring_mechanism_sender = None

            # Check if error_source has a projection to a MonitoringMechanism or a ProcessingMechanism
            for projection in error_source.outputState.sendsToProjections:
                if isinstance(projection.receiver.owner, MonitoringMechanism):
                    # If projection to MonitoringMechanism is found, assign and quit search
                    monitoring_mechanism = projection.receiver.owner
                    break
                # IMPLEMENTATION NOTE:
                #    the following finds only the last or only projection to a ProcessingMechanism with a LearningSignal
                if isinstance(projection.receiver.owner, ProcessingMechanism):
                    try:
                        next_level_learning_signal = projection.parameterStates[kwMatrix]
                    except:
                        pass
                    else:
                        next_level_monitoring_mechanism_sender = next_level_learning_signal.sender
                        next_level_weight_matrix = projection.paramsCurrent[kwExecuteMethod][kwMatrix]

            # error_source does not project to a MonitoringMechanism
            if not monitoring_mechanism:

                # error_source DOES project to a ProcessingMechanism:
                #    instantiate WeightedError MonitoringMechanism:
                #        computes contribution of each element in error_source to error at level to which it projects
                if next_level_monitoring_mechanism_sender:
                    error_source_output = np.zeros_like(error_source.outputState.value)
                    monitoring_mechanism = WeightedError(error_signal=error_source_output,
                                                         params={kwMatrix:next_level_weight_matrix})

                # error_source does NOT project to a ProcessingMechanism:
                #     instantiate DefaultTrainingMechanism MonitoringMechanism
                #         (compares error_source output with external training signal)
                else:
                    output_signal = np.zeros_like(error_source.outputState.value)
                    # IMPLEMENTATION NOTE: training_signal assigment currently assumes training mech is LinearComparator
                    training_signal = output_signal
                    training_mechanism_input = np.array([output_signal, training_signal])
                    monitoring_mechanism = DefaultTrainingMechanism(training_mechanism_input)
                    # Instantiate a mapping projection from the error_source to the DefaultTrainingMechanism
                    Mapping(sender=error_source, receiver=monitoring_mechanism)

            self.sender = monitoring_mechanism.outputState
            # Add self as outgoing projection from MonitoringMechanism
            from PsyNeuLink.Functions.Projections.Projection import add_projection_to
            add_projection_from(sender=monitoring_mechanism,
                                state=monitoring_mechanism.outputState,
                                projection_spec=self,
                                receiver=self.receiver,
                                context=context)

        # IMPLEMENTATION NOTE:  MOVED FROM instantiate_receiver
        # Insure that LearningSignal output (error signal) and receiver's weight matrix are same shape
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


        # IMPLEMENTATION NOTE:  MOVED FROM instantiate_receiver
        # Add LearningSignal projection to receiver's parameterState
        # self.add_to(receiver=self.receiver, state=ParameterState, context=context)
        self.add_to(receiver=self.receiver.owner, state=self.receiver, context=context)

# NO SENDER SO:
    # error_source PROJECTS TO A MONITORING MECHANISM
    #         assign it as sender
    # error_source DOESN'T PROJECT TO A MONITORING MECHANISM
        # error_source PROJECTS DOESN'T PROJECT TO A PROCESSING MECHANISM:
            # INSTANTIATE DefaultTrainingMechanism
        # error_source PROJECTS DOES PROJECT TO A PROCESSING MECHANISM:
            # INSTANTIATE WeightedSum MonitoringMechanism



            # elif:
            #     # IMPLEMENT: ASSIGN??/CHECK FOR?? error_source.errorSignal AND ASSIGN TO self.variable
            #     pass
            # else:
            #     # IMPLEMENT: RAISE EXCEPTION FOR MISSING MONITORING MECHANISM / SOURCE OF ERROR SIGNAL FOR LEARNING SIGNAL
            #     #            OR INSTANTIATE DEFAULT MONITORING MECHANISM
            #     pass
            #
            #     # FIX: ??CALL:
            #     # super().instantiate_sender(context=context)

    def update(self, params=NotImplemented, context=NotImplemented):
        """

        DOCUMENT:
        LearnningSignal (Projection):
            - sender:  output of Monitoring Mechanism
                default: receiver.owner.outputState.sendsToProjections.<MonitoringMechanism> if specified,
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
        # Array of output values for Mapping projection's receiver mechanism
        # LearningSignal(self).ParameterState(receiver).Mapping(owner).OutputState(receiver).ProcessMechanism(owner)
        output = self.receiver.owner.receiver.owner.value

        # ASSIGN ERROR
        error_signal = self.variable

        # CALL EXECUTE METHOD TO GET WEIGHT CHANGES
        # rows:  sender errors;  columns:  receiver errors
        self.weightChangeMatrix = self.execute([input, output, error_signal], params=params, context=context)

        # # Sum rows of weightChangeMatrix to get errors for each item of Mapping projection's sender
        # self.weightChanges = np.add.reduce(self.weightChangeMatrix,1)

        return self.weightChangeMatrix
