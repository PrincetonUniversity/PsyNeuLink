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
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.Comparator import Comparator
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.WeightedError import WeightedError
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms import ProcessingMechanism
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base

# from Functions.Utility import *

# Params:

kwWeightChangeParams = "Weight Change Params"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

DefaultTrainingMechanism = Comparator

class LearningSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningSignal(Projection_Base):
# DOCUMENT: USES DEFERRED INITIALIZATION
# DOCUMENT: self.variable has 3 items:
#               input: <Mapping projection>.sender.value
#                          (output of the Mechanism that is the sender of the Mapping projection)
#               output: <Mapping projection>.receiver.owner.outputState.value == self.errorSource.outputState.value
#                          (output of the Mechanism that is the receiver of the Mapping projection)
#               error_signal: <Mapping projection>.receiver.owner.parameterState[kwMatrix].receivesFromProjections[0].sender.value ==
#                                 self.errorSource.monitoringMechanism.value
#                                 (output of the MonitoringMechanism that is the sender of the LearningSignal for the next Mapping projection in the Process)
# DOCUMENT: if it instantiates a DefaultTrainingSignal:
#               if outputState for error source is specified in its paramsCurrent[kwMonitorForLearning], use that
#               otherwise, use error_soure.outputState (i.e., error source's primary outputState)

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
            + kwFunction (Utility): (default: BP)
            + kwFunctionParams (dict):
                + kwLearningRate (value): (default: 1)
        - name (str) - if it is not specified, a default based on the class is assigned in register_category
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)

    Parameters:
        The default for kwFunction is BackPropagation:
        The parameters of kwFunction can be set:
            - by including them at initialization (param[kwFunction] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwFunction].adjust(params)
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
            + kwFunction (Utility): (default: BP)
            + kwFunctionParams:
                + kwLearningRate (value): (default: 1)
        + paramNames (dict)
        + classPreference (PreferenceSet): LearningSignalPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

    Class methods:
        function (executes function specified in params[kwFunction]

    Instance attributes:
        + sender (MonitoringMechanism)
        + receiver (Mapping)
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramsCurrent (dict) - set currently in effect
        + variable (value) - used as input to projection's execute method
        + value (value) - output of execute method
        + mappingWeightMatrix (2D np.array) - points to <Mapping>.paramsCurrent[kwFunctionParams][kwMatrix]
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

    # variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwProjectionSender: MonitoringMechanism_Base,
                               kwFunction:BackPropagation,
                               kwFunctionParams: {BackPropagation.kwLearningRate: 1,
                                                       kwParameterStates: None # This suppresses parameterStates
                                                       },
                               kwWeightChangeParams: {
                                   kwFunction: LinearCombination,
                                   kwFunctionParams: {kwOperation: LinearCombination.Operation.SUM},
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

        # self.sender_arg = sender
        # self.receiver_arg = receiver
        # self.params_arg = params
        # self.prefs_arg = prefs

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        # MODIFIED 8/14/16 OLD:
        # Store args for deferred initialization
        self.init_args = locals().copy()
        self.init_args['context'] = self
        self.init_args['name'] = name
        del self.init_args['self']
        # del self.init_args['__class__']

        # Flag for deferred initialization
        self.value = kwDeferredInit

        # # MODIFIED 8/14/16 NEW:
        # # PROBLEM: variable has different name for different classes; need to standardize across classes
        # context = self
        # name = self.name
        # super().__init__(sender=sender,
        #                  receiver=receiver,
        #                  params=params,
        #                  name=name,
        #                  prefs=prefs,
        #                  context=context)


    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Insure sender is a MonitoringMechanism or ProcessingMechanism and receiver is a ParameterState or Mapping

        Validate send in params[kwProjectionSender] or, if not specified, sender arg:
        - must be the outputState of a MonitoringMechanism (e.g., Comparator or WeightedError)
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

        # If specification is a MonitoringMechanism class, pass (it will be instantiated in instantiate_sender)
        elif inspect.isclass(sender) and issubclass(sender,  MonitoringMechanism_Base):
            pass

        else:
            raise LearningSignalError("Sender arg (or {} param ({}) for must be a MonitoringMechanism, "
                                      "the outputState of one, or a reference to the class"
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
            reference the Mapping projection's weight matrix: self.mappingProjection.matrix

        """
        # FIX: PROBLEM: instantiate_receiver usually follows instantiate_execute_method,
        # FIX:          and uses self.value (output of execute method) to validate against receiver.variable

        self.instantiate_receiver(context)

        # # MODIFIED 8/14/16: COMMENTED OUT SINCE SOLVED BY MOVING add_to TO instantiate_attributes_after_execute_method
        # # "Cast" self.value to Mapping Projection parameterState's variable to pass validation in instantiate_sender
        # # Note: this is because instantiate_sender calls add_projection_to
        # # (since self.value is not assigned until instantiate_execute_method; it will be reassigned there)
        # self.value = self.receiver.variable

        super().instantiate_attributes_before_execute_method(context)

    def instantiate_attributes_after_execute_method(self, context=NotImplemented):
        """Override super since it calls instantiate_receiver which has already been called above
        """
        # pass
        # MODIFIED 8/14/16: MOVED FROM instantiate_sender
        # Add LearningSignal projection to Mapping projection's parameterState
        # Note: needs to be done after instantiate_execute_method, since validation requires self.value be assigned
        self.add_to(receiver=self.mappingProjection, state=self.receiver, context=context)

    def instantiate_receiver(self, context=NotImplemented):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a Mapping Projection, assign LearningSignal to parameterStates[kwMatrix]
            for the projection;  if that does not exist, instantiate and assign as the receiver for the LearningSignal
        If specified as a ParameterState, validate that it is parameterStates[kwMatrix]
        Validate that the LearningSignal's error matrix is the same shape as the recevier's weight matrix
        Re-assign LearningSignal's variable to match the height (number of rows) of the matrix
        
        Notes:
        * This must be called before instantiate_sender since that requires access to self.receiver
            to determine whether to use a comparator mechanism or <Mapping>.receiverError for error signals
        * Doesn't call super().instantiate_receiver since that assumes self.receiver.owner is a Mechanism
                              and calls add_projection_to_mechanism
        """

# FIX: ??REINSTATE CALL TO SUPER AFTER GENERALIZING IT TO USE Projection.add_to
# FIX: OR, MAKE SURE FUNCTIONALITY IS COMPARABLE

        weight_change_params = self.paramsCurrent[kwWeightChangeParams]

        # VALIDATE that self.receiver is a ParameterState or a Mapping Projection

        # If receiver is a ParameterState, and receiver's parameterStates dict has been instantiated,
        #    make sure LearningSignal is being assigned to the parameterStates[kwMatrix] of a Mapping projection
        if isinstance(self.receiver, ParameterState):

            self.mappingProjection = self.receiver.owner

            if not isinstance(self.mappingProjection, Mapping):
                raise LearningSignalError("Receiver arg ({}) for {} must be the "
                                          "parameterStates[{}] of a Mapping (rather than a {}) projection".
                                          format(self.receiver,
                                                 self.name,
                                                 kwMatrix,
                                                 self.mappingProjection.__class__.__name__))

            # receiver is parameterState[kwMatrix], so update its params with ones specified by LearningSignal
            if (self.mappingProjection.parameterStates and
                    self.receiver is self.mappingProjection.parameterStates[kwMatrix]):
                # FIX: ?? SHOULD THIS USE assign_defaults:
                self.receiver.paramsCurrent.update(weight_change_params)

            else:
                raise LearningSignalError("Receiver arg ({}) for {} must be the "
                                          "parameterStates[{}] param of the receiver".
                                          format(self.receiver, self.name, kwMatrix))

        # Receiver was specified as a Mapping Projection
        elif isinstance(self.receiver, Mapping):

            self.mappingProjection = self.receiver

            from PsyNeuLink.Functions.States.InputState import instantiate_state_list
            from PsyNeuLink.Functions.States.InputState import instantiate_state

            # Check if Mapping Projection has parameterStates Ordered Dict and kwMatrix entry
            try:
                self.receiver.parameterStates[kwMatrix]

            # receiver does NOT have parameterStates attrib
            except AttributeError:
                # Instantiate parameterStates Ordered dict
                #     with ParameterState for receiver's functionParams[kwMatrix] param
                self.receiver.parameterStates = instantiate_state_list(owner=self.receiver,
                                                                       state_list=[(kwMatrix,
                                                                                    weight_change_params)],
                                                                       state_type=ParameterState,
                                                                       state_param_identifier=kwParameterState,
                                                                       constraint_value=self.mappingWeightMatrix,
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
                                                                            constraint_value=self.mappingWeightMatrix,
                                                                            constraint_value_name=kwLearningSignal,
                                                                            context=context)

            # receiver has parameterState for kwMatrix, so update its params with ones specified by LearningSignal
            else:
                # MODIFIED 8/13/16:
                # FIX: ?? SHOULD THIS USE assign_defaults:
                self.receiver.parameterStates[kwMatrix].paramsCurrent.update(weight_change_params)

            # Assign self.receiver to parameterState used for weight matrix param
            self.receiver = self.receiver.parameterStates[kwMatrix]

        # If it is not a ParameterState or Mapping Projection, raise exception
        else:
            raise LearningSignalError("Receiver arg ({}) for {} must be a Mapping projection or"
                                      " a MechanismParatemerState of one".format(self.receiver, self.name))

        # GET RECEIVER'S WEIGHT MATRIX
        self.get_mapping_projection_weight_matrix()

        # Format input to Mapping projection's weight matrix
        # MODIFIED 8/19/16:
        # self.input_to_weight_matrix = np.zeros_like(self.mappingWeightMatrix[0])
        self.input_to_weight_matrix = np.zeros_like(self.mappingWeightMatrix.T[0])

        # Format output of Mapping projection's weight matrix
        # Note: this is used as a template for output value of its receiver mechanism (i.e., to which it projects)
        #       but that may not yet have been instantiated;  assumes that format of input = output for receiver mech
        # MODIFIED 8/19/16:
        # self.output_of_weight_matrix = np.zeros_like(self.mappingWeightMatrix.T[0])
        self.output_of_weight_matrix = np.zeros_like(self.mappingWeightMatrix[0])

    def get_mapping_projection_weight_matrix(self):
        """Get weight matrix for Mapping projection to which LearningSignal projects

        """

        message = "PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".format(self.receiver.name,
                                                                                             kwFunctionParams,
                                                                                             kwMatrix)
        if isinstance(self.receiver, ParameterState):
            try:
                self.mappingWeightMatrix = self.mappingProjection.matrix
            except KeyError:
                raise LearningSignal(message)

        elif isinstance(self.receiver, Mapping):
            try:
                self.mappingWeightMatrix = self.receiver.matrix
            except KeyError:
                raise LearningSignal(message)


    def instantiate_sender(self, context=NotImplemented):
        # DOCUMENT: SEE UPDATE BELOW
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
           UPDATE: otherwise, if Mapping projection's receiver has an errorSignal attribute, use that as self.variable
               (e.g., "hidden units in a multilayered neural network, using BackPropagation Function)
           [TBI: otherwise, implement default MonitoringMechanism]
           otherwise, raise exception

FROM TODO:
#    - instantiate_sender:
#        - examine mechanism to which Mapping projection projects:  self.receiver.owner.receiver.owner
#            - check if it is a terminal mechanism in the system:
#                - if so, assign:
#                    - Comparator MonitoringMechanism
#                        - ProcessInputState for Comparator (name it??) with projection to target inputState
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
    # self.errorSource PROJECTS TO A MONITORING MECHANISM
    #         assign it as sender
    # self.errorSource DOESN'T PROJECT TO A MONITORING MECHANISM
        # self.errorSource PROJECTS TO A PROCESSING MECHANISM:
            # INSTANTIATE WeightedSum MonitoringMechanism
        # self.errorSource PROJECTS DOESN'T PROJECT TO A PROCESSING MECHANISM:
            # INSTANTIATE DefaultTrainingMechanism

        """

        # FIX: 8/7/16
        # FIX: NEED TO DEAL HERE WITH CLASS SPECIFICATION OF MonitoringMechanism AS DEFAULT
        # FIX: OR HAVE ALREADY INSTANTIATED DEFAULT MONITORING MECHANISM BEFORE REACHING HERE
        # FIX: EMULATE HANDLING OF DefaultMechanism (for Mapping) AND DefaultController (for ControlSignal)

        # FIX: 8/18/16
        # FIX: ****************
        # FIX: ASSIGN monitoring_source IN ifS, NOT JUST else
        # FIX: SAME FOR self.errorSource??

        monitoring_mechanism = None

        # MonitoringMechanism specified for sender
        if isinstance(self.sender, MonitoringMechanism_Base):
            # Re-assign to outputState
            self.sender = self.sender.outputState

        # OutputState specified for sender
        if isinstance(self.sender, OutputState):
            # - validate that it belongs to a MonitoringMechanism
            if not isinstance(self.sender.owner, MonitoringMechanism_Base):
                raise LearningSignalError("OutputState ({}) specified as sender for {} belongs to a {}"
                                          " rather than a MonitoringMechanism".
                                          format(self.sender.name,
                                                 self.name,
                                                 self.sender.owner.__class__.__name__))
            # - validate that the length of the sender (MonitoringMechanism)'s outputState.value (the error signal)
            #     is the same as the width (# columns) of Mapping projection's weight matrix (# of receivers)
            # - assign MonitoringMechanism's outputState.value as self.variable
            if len(self.sender.value) == len(self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]):
                self.errorSource = self.mappingProjection.receiver.owner
            else:
                raise LearningSignalError("Length ({}) of MonitoringMechanism outputState specified as sender for {} "
                                          "must match the receiver dimension ({}) of the weight matrix for {}".
                                          format(len(self.sender.outputState.value),
                                                 self.name,
                                                 len(self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]),
                                                 # self.receiver.owner))
                                                 self.mappingProjection))
            # Add reference to MonitoringMechanism to Mapping projection
            monitoring_mechanism = self.sender

        # MonitoringMechanism class (i.e., not an instantiated object) specified for sender, so instantiate it:
        # - for terminal mechanism of Process, instantiate Comparator MonitoringMechanism
        # - for preceding mechanisms, instantiate WeightedSum MonitoringMechanism
        else:
            # Get errorSource:  ProcessingMechanism for which error is being monitored
            #    (i.e., the mechanism to which the Mapping projection projects)
            # Note: Mapping.instantiate_receiver has not yet been called, so need to do parse below
            from PsyNeuLink.Functions.States.InputState import InputState
            if isinstance(self.mappingProjection.receiver, Mechanism):
                self.errorSource = self.mappingProjection.receiver
            elif isinstance(self.mappingProjection.receiver, InputState):
                self.errorSource = self.mappingProjection.receiver.owner

            next_level_monitoring_mechanism = None

            # Check if errorSource has a projection to a MonitoringMechanism or a ProcessingMechanism
            for projection in self.errorSource.outputState.sendsToProjections:
                # errorSource has a projection to a MonitoringMechanism, so assign it and quit search
                if isinstance(projection.receiver.owner, MonitoringMechanism_Base):
                    monitoring_mechanism = projection.receiver.owner
                    break
                # errorSource has a projection to a ProcessingMechanism, so determine whether that has a LearningSignal
                if isinstance(projection.receiver.owner, ProcessingMechanism_Base):
                    try:
                        next_level_learning_signal = projection.parameterStates[kwMatrix].receivesFromProjections[0]
                    except (AttributeError, KeyError):
                        # Next level's projection has no parameterStates, Matrix parameterState or projections to it
                        #    => no LearningSignal
                        pass
                    else:
                        # Next level's projection has a LearningSignal so get:
                        #     the weight matrix for the next level's projection
                        #     the MonitoringMechanism that provides error_signal
                        next_level_weight_matrix = projection.matrix
                        next_level_monitoring_mechanism = next_level_learning_signal.sender

            # errorSource does not project to a MonitoringMechanism
            if not monitoring_mechanism:

                # NON-TERMINAL Mechanism
                # errorSource DOES project to a MonitoringMechanism:
                #    instantiate WeightedError MonitoringMechanism and the back-projection for its error signal:
                #        (computes contribution of each element in errorSource to error at level to which it projects)
                if next_level_monitoring_mechanism:
                    error_signal = np.zeros_like(next_level_monitoring_mechanism.value)
                    monitoring_mechanism = WeightedError(error_signal=error_signal,
                                                         params={kwMatrix:next_level_weight_matrix})

                    # Instantiate mapping projection to provide monitoring_mechanism with error signal
                    Mapping(sender=next_level_monitoring_mechanism,
                            receiver=monitoring_mechanism,
                            # name=monitoring_mechanism.name+'_'+kwMapping)
                            name=next_level_monitoring_mechanism.name +
                                 ' to '+monitoring_mechanism.name +
                                 ' ' + kwMapping + ' Projection')

                # TERMINAL Mechanism
                # errorSource does NOT project to a MonitoringMechanism:
                #     instantiate DefaultTrainingMechanism MonitoringMechanism
                #         (compares errorSource output with external training signal)
                else:
                    output_signal = np.zeros_like(self.errorSource.outputState.value)
                    # IMPLEMENTATION NOTE: training_signal assigment currently assumes training mech is Comparator
                    training_signal = output_signal
                    training_mechanism_input = np.array([output_signal, training_signal])
                    monitoring_mechanism = DefaultTrainingMechanism(training_mechanism_input)
                    # Instantiate a mapping projection from the errorSource to the DefaultTrainingMechanism
                    try:
                        monitored_state = self.errorSource.paramsCurrent[kwMonitorForLearning]
                        monitored_state = self.errorSource.outputStates[monitored_state]
                    except KeyError:
                        # No state specified so use Mechanism as sender arg
                        monitored_state = self.errorSource
                    Mapping(sender=monitored_state,
                            receiver=monitoring_mechanism,
                            name=self.errorSource.name+' to '+monitoring_mechanism.name+' '+kwMapping+' Projection')

            self.sender = monitoring_mechanism.outputState

            # "Cast" self.variable to match value of sender (MonitoringMechanism) to pass validation in add_to()
            # Note: self.variable will be re-assigned in instantiate_execute_method()
            self.variable = self.errorSignal

            # Add self as outgoing projection from MonitoringMechanism
            from PsyNeuLink.Functions.Projections.Projection import add_projection_to
            add_projection_from(sender=monitoring_mechanism,
                                state=monitoring_mechanism.outputState,
                                projection_spec=self,
                                receiver=self.receiver,
                                context=context)

        # Add reference to MonitoringMechanism to Mapping projection
        self.mappingProjection.monitoringMechanism = monitoring_mechanism

    def instantiate_execute_method(self, context=NotImplemented):
        """Construct self.variable for input to function, call super to instantiate it, and validate output

        function implements function to compute weight change matrix for receiver (Mapping projection) from:
        - input: array of sender values (rows) to Mapping weight matrix (self.variable[0])
        - output: array of receiver values (cols) for Mapping weight matrix (self.variable[1])
        - error:  array of error signals for receiver values (self.variable[2])
        """

        # Reconstruct self.variable as input for function
        self.variable = [[0]] * 3
        self.variable[0] = self.input_to_weight_matrix
        self.variable[1] = self.output_of_weight_matrix
        self.variable[2] = self.errorSignal

        super().instantiate_execute_method(context)

        # FIX: MOVE TO AFTER INSTANTIATE FUNCTION??
        # IMPLEMENTATION NOTE:  MOVED FROM instantiate_receiver
        # Insure that LearningSignal output (error signal) and receiver's weight matrix are same shape
        try:
            receiver_weight_matrix_shape = self.mappingWeightMatrix.shape
        except TypeError:
            # self.mappingWeightMatrix = 1
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
                                         # self.receiver.owner.name))
                                         self.mappingProjection.name))



    def update(self, params=NotImplemented, time_scale=NotImplemented, context=NotImplemented):
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
        - errorDerivative:  get from kwFunction of Comparator Mechanism
        - transferDerivative:  get from kwFunction of Processing Mechanism

        :return: (2D np.array) self.weightChangeMatrix
        """
        # Pass during initialization (since has not yet been fully initialized
        if self.value is kwDeferredInit:
            return self.value

        # GET INPUT TO Projection to Error Source:
        # Array of input values from Mapping projection's sender mechanism's outputState
# FIX: IMPLEMENT self.unconvertedInput AND self.convertedInput, VALIDATE QUANTITY BELOW IN instantiate_sender, ASSIGN self.input ACCORDINGLY
        input = self.mappingProjection.sender.value

        # ASSIGN OUTPUT TO ERROR SOURCE
        # Array of output values for Mapping projection's receiver mechanism
        # output = self.mappingProjection.receiver.owner.outputState.value
# FIX: IMPLEMENT self.unconvertedOutput AND self.convertedOutput, VALIDATE QUANTITY BELOW IN instantiate_sender, ASSIGN self.input ACCORDINGLY
        output = self.errorSource.outputState.value

        # ASSIGN ERROR
# FIX: IMPLEMENT self.input AND self.convertedInput, VALIDATE QUANTITY BELOW IN instantiate_sender, ASSIGN ACCORDINGLY
        error_signal = self.errorSignal

        # CALL EXECUTE METHOD TO GET WEIGHT CHANGES
        # rows:  sender errors;  columns:  receiver errors
# FIX: self.weightChangeMatrix = self.execute([self.input, self.output, self.error_signal], params=params, context=context)
        self.weightChangeMatrix = self.execute([input, output, error_signal], params=params, context=context)

        if not kwInit in context:
            print("\n{} Weight Change Matrix: \n{}\n".format(self.name, self.weightChangeMatrix))

        self.value = self.weightChangeMatrix

        return self.value

    @property
    def errorSignal(self):
        return self.sender.value
        