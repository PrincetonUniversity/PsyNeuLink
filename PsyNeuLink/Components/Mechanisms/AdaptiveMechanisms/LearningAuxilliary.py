# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningAuxilliary ******************************************************

"""
.. _LearningAuxilliary_Overview:

Overview
--------

This module contains functions that automatically construct and compose the components necessary for learning
when this is specified for a process.

COMMENT:
    IMPLEMENT: LearningMechanism:
        PROCESS & SYSTEM:
          • Convert ProcessInputState and SystemInputState into Mechanisms with LinearFunction IDENTITY_FUNCTION
          • Use only one ObjectiveMechanism for all levels with the following args:
                default_input_value[[ACTIVITY][ERROR]]
                monitored_values: [[next_level_mech.OutputState][next_level_mech.objective_mechanism.OutputState]]
                names: [[ACTIVITY][ERROR]]
                function:  ErrorDerivative(variable, derivative)
                               variable[0] = activity
                               variable[1] = error_signal from next_level_mech ObjectiveMechanism (target for TERMINAL)
                               derivative = error_derivative (1 for TERMINAL)
                role:LEARNING
          • Use only one Learning mechanism with the following args:
                variable[[ACTIVATION_INPUT_INDEX][ACTIVATION_SAMPLE_INDEX][ERROR_SIGNAL_INDEX]
                activation_derivative
                error_matrix
                function
            Initialize and assign args with the following WIZZARD:
        WIZZARD:
            Needs to know
                activation_sample_mech (Mechanism)
                    activation_derivative (function)
                next_level_mech (Mechanism)
                    error_derivative (function)
                    error_matrix (ndarray) - for MappingProjection from activation_sample_mech to next_level_mech
            ObjectiveMechanism:
                Initialize variable:
                      use next_level_mech.outputState.valuee to initialize variable[ACTIVITY]
                      use outputState.value of next_level_mech's objective_mechanism to initialize variable[ERROR]
                Assign mapping projections:
                      nextLevel.outputState.value -> inputStates[ACTIVITY] of ObjectiveMechanism
                      nextLevel.objective_mechanism.outputState.value  -> inputStates[ERROR] of ObjectiveMechanism
                NOTE: For TERMINAL mechanism:
                          next_level_mech is Process or System InputState (function=Linear, so derivative =1), so that
                             next_level_mech.outputState.value is the target, and
                             error_derivative = 1
                             error_matrix = IDENTITY_MATRIX (this should be imposed)
            LearningMechanism:
                Initialize variable:
                      use mapping_projection.sender.value to initialize variable[ACTIVATION_INPUT_INDEX]
                      use activation_sample_mech_outputState.value to initialize variable[ACTIVATION_SAMPLE_INDEX]
                      use next_level_mech.objecdtive_mechanism.OutputState.value to initialize variable[ERROR_SIGNAL_INDEX]
                Assign activation_derivative using function of activation_sample_mech of mapping_projection (one being learned)
                Assign error_derivative using function of next_level_mech
                Assign error_matrix as runtime_param using projection to next_level_mech [ALT: ADD TO VARIABLE]
                Assign mapping projections:
                      mapping_projection.sender -> inputStates[ACTIVATION_INPUT_INDEX] of LearningMechanism
                      activation_sample_mech.outputState -> inputStates[ACTIVATION_SAMPLE_INDEX] of LearningMechanism
                      next_level_mech.objective_mechanism.OutputState.value -> inputStates[ERROR_SIGNAL_INDEX]

            For TARGET MECHANISM:  Matrix is IDENTITY MATRIX??
            For TARGET MECHANISM:  derivative for ObjectiveMechanism IDENTITY FUNCTION
COMMENT


.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

def _is_learning_spec(spec):
    """Evaluate whether spec is a valid learning specification

    Return :keyword:`true` if spec is LEARNING or a valid projection_spec (see Projection._is_projection_spec
    Otherwise, return :keyword:`False`

    """
    if spec is LEARNING:
        return True
    else:
        return _is_projection_spec(spec)


    # mappingProjection : MappingProjection
    #     the `MappingProjection` that owns the `parameterState <ParameterState>` to which the
    #     LearningProjection projects (i.e., that owns its `receiver <LearningProjection.receiver>`).
    #
    # errorSource : ProcessingMechanism
    #     the mechanism to which `mappingProjection` projects, and that is used to calculate the `error_signal`.


# def objective_mechanism_wizzard(recipient:Mechanism,
#                              objective_mechanism_spec:tc.optional(Mechanism, OutputState, ObjectiveMechanism)=None,
#                              output_spec=None
#                              ):
def instantiate_learning(learning_projection:LearningProjection):
    """
    Call in _instantiate_attributes_before_function() of LearningProjection
    Do the following:
        Get:
            activation_projection (one being learned) (MappingProjection)
            activation_mech (ProcessingMechanism)
            activation_input (OutputState)
            activation_sample (OutputState)
            activation_derivative (function)
            error_matrix (ParameterState)
            error_derivative (function)
            next_level_mech (error_source_mech) (ProcessingMechanism)
            next_level_objective_mech (error_source_objective_mech) (ObjectiveMechanism)
        Instantiate:
            ObjectiveMechanism:
                Construct with:
                   monitor_values: [next_level_mech.outputState.value, next_level_mech.objective_mech.outputState.value]
                   names = [ACTIVITY/SAMPLE, ERROR/TARGET]
                   function = ErrorDerivative(derivative=error_derivative)
                NOTE: For TERMINAL mechanism:
                          next_level_mech is Process or System InputState (function=Linear, so derivative =1), so that
                             next_level_mech.outputState.value is the target, and
                             error_derivative = 1
                             error_matrix = IDENTITY_MATRIX (this should be imposed)
            LearningMechanism:
                Construct with:
                    variable=[[activation_input],[activation_sample],[ObjectiveMechanism.outputState.value]]
                    error_matrix = error_matrix
                    function = one specified with Learning specification
                               NOTE: can no longer be specified as function for LearningProjection
                    names = [ACTIVATION_INPUT, ACTIVATION_SAMPLE, ERROR_SIGNAL]
                        NOTE:  this needs to be implemented for LearningMechanism as it is for ObjectiveMechanism
                Check that, if learning function expects a derivative (in user_params), that the one specified
                    is compatible with the function of the activation_mech
                Assign:
                    NOTE:  should do these in their own Learning module function, called by LearningMechanaism directly
                        as is done for ObjectiveMechanism
                    MappingProjection: activation_projection.sender -> LearningMechanism.inputState[ACTIVATION_INPUT]
                    MappingProjection: activation_mech.outputState -> LearningMechanism.inputState[ACTIVATION_SAMPLE]
                    MappingProjection: ObjectiveMechanism -> LearningMechanism.inputState[ERROR_SIGNAL]

    ----------------------------------------------------------------------



    Parse `objective_mechanism_spec` specification and call for implementation if necessary, including a
        `MappingProjection` from it to the recipient's `primary inputState <Mechanism_InputStates>`.
    Assign its outputState to _objective_mechanism_output.
    Verify that outputState's value is compatible with `error_signal`.


    FROM LearningProjection:  [STILL NEEDED??]
    Call _instantiate_receiver first since both _instantiate_objective_mechanism and _instantiate_function
    reference the receiver's (i.e., MappingProjection's) weight matrix: self.mappingProjection.matrix

    """

    # Get:
    #     activation_projection (one being learned) (MappingProjection)
    #     activation_mech (ProcessingMechanism)
    #     activation_input (OutputState)
    #     activation_sample (OutputState)
    #     activation_derivative (function)
    #     error_matrix (ParameterState)
    #     error_derivative (function)
    #     next_level_mech (error_source_mech) (ProcessingMechanism)
    #     next_level_objective_mech (error_source_objective_mech) (ObjectiveMechanism)

    # Get projection being learned:
    activation_projection = learning_projection


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # FIX: PROBLEM: _instantiate_receiver usually follows _instantiate_function,
    # FIX:          and uses self.value (output of function) to validate against receiver.variable
    self._instantiate_receiver(context)

    super()._instantiate_attributes_before_function(context)


    # Do stuff in instantiate_receiver of LearningProjection:  instantiate LearningMechanism
    #
    # Override super to call _instantiate_receiver before calling _instantiate_sender

    # ----------------------------------------------------------------------------------------------------------
    # NEW STUFF ------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    # IMPLEMENTION NOTE:  REDUNDANT WITH typecheck?
    # Validate objective_mechanism_spec
    if objective_mech_spec and not any(m in {None, ObjectiveMechanism, OutputState, Mechanism} for
                                       m in {objective_mech_spec, type(objective_mech_spec)}):
        raise LearningMechanismError("Specification for {} arg of {} must ({}) must be "
                                     "a Mechanism, OutputState or \`ObjectiveMechanism\`".
                                     format(OBJECTIVE_MECHANISM, self.name, target_set[OBJECTIVE_MECHANISM]))

    # If objective_mechanism_spec is not specified, defer to Composition for instantiation
    if objective_mechanism_spec is None:
        # FIX: RETURN HERE?  HOW TO HANDLE NON-INSTANTIATED _objective_mechanism_output?
        pass
    # If objective_mechanism_spec is specified by class, call module method to instantiate one
    # IMPLEMENTATION NOTE:  THIS SHOULD BE HANDLED BY Composition ONCE IT IS IMPLEMENTED
    elif objective_mechanism_spec is ObjectiveMechanism:
        objective_mechanism_spec = _instantiate_objective_mechanism(self, context=context)

    else:
        raise LearningMechanismError("PROGRAM ERROR: invalid type for objective_mechanism_spec pass validation")

    objective_mechanism_output = None

    if objective_mechanism_spec:
        # If _objective_mechanism_output is already an outputState, assign it to _objective_mechanism_output
        if isinstance(objective_mechanism_spec, OutputState):
            objective_mechanism_output = objective_mechanism_spec

        # If objective_mechanism_spec is specified as a Mechanism,
        #    assign _objective_mechanism_output to the mechanism's primary OutputState
        if isinstance(objective_mechanism_spec, Mechanism):
            objective_mechanism_output = objective_mechanism_spec.outputState

        if not objective_mechanism_output:
            raise LearningMechanismError("PROGRAMM ERROR: objective_mechanism_spec requested for {} not recognized ".
                                         format(recipient.name))

        # Validate that _objective_mechanism_output is a 1D np.array
        if not isinstance(objective_mechanism_output, (list, np.ndarray)):
            raise LearningMechanismError("The output of the objective_mechanism_spec for {} must be a list or 1D array".
                                         format(self.name, sender))
        if not np.array(objective_mechanism_output).ndim == 1:
            raise LearningMechanismError("The output of the objective_mechanism_spec for {} must be an 1D array".
                                      format(self.name, self.name))

        # Validate that _objective_mechanism_output matches format of error_signal
        if not iscompatible(self.error_signal, objective_mechanism_output.value):
            raise LearningMechanismError("The output ({}) of objective_mechanism_spec ({}) must match the "
                                         "error_signal {} for {} in its length and type of elements".
                                         format(objective_mechanism_output.value,
                                                objective_mechanism_spec,
                                                self.error_signal,
                                                self.name))

        # Validate that there is a MappingProjection from objective_mechanism_spec
        #    to the LearningMechanism's primary inputState
        if not any(objective_mechanism_spec.output is p for p in self.inputState.receivesFromProjections):
            raise LearningMechanismError("{} does not have a MappingProjection from "
                                         "its specified objective_mechanism_spec ({})".
                                         format(self.name, objective_mechanism_spec.name))

        return objective_mechanism

    # ----------------------------------------------------------------------------------------------------------
    # END NEW STUFF --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------


    def _instantiate_attributes_after_function(self, context=None):
        """Override super since it calls _instantiate_receiver which has already been called above
        """
        # pass
        # MODIFIED 8/14/16: MOVED FROM _instantiate_sender
        # Add LearningProjection to MappingProjection's parameterState
        # Note: needs to be done after _instantiate_function, since validation requires self.value be assigned
        self.add_to(receiver=self.mappingProjection, state=self.receiver, context=context)


    def _instantiate_receiver(self, context=None):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a MappingProjection, assign LearningProjection to parameterStates[MATRIX]
            for the projection;  if that does not exist, instantiate and assign as the receiver for the LearningProjection
        If specified as a ParameterState, validate that it is parameterStates[MATRIX]
        Validate that the LearningProjection's error matrix is the same shape as the recevier's weight matrix
        Re-assign LearningProjection's variable to match the height (number of rows) of the matrix

        Notes:
        * This must be called before _instantiate_sender since that requires access to self.receiver
            to determine whether to use a ComparatorMechanism or <MappingProjection>.receiverError for error signals
        * Doesn't call super()._instantiate_receiver since that assumes self.receiver.owner is a Mechanism
                              and calls _add_projection_to_mechanism
        """

# FIX: ??REINSTATE CALL TO SUPER AFTER GENERALIZING IT TO USE Projection.add_to
# FIX: OR, MAKE SURE FUNCTIONALITY IS COMPARABLE

        weight_change_params = self.paramsCurrent[WEIGHT_CHANGE_PARAMS]

        # VALIDATE that self.receiver is a ParameterState or a MappingProjection

        # If receiver is a ParameterState, and receiver's parameterStates dict has been instantiated,
        #    make sure LearningProjection is being assigned to the parameterStates[MATRIX] of a MappingProjection
        if isinstance(self.receiver, ParameterState):

            self.mappingProjection = self.receiver.owner

            # MODIFIED 10/29/16 OLD:
            # Reciever must be a MappingProjection with a LinearCombination function
            if not isinstance(self.mappingProjection, MappingProjection):
                raise LearningProjectionError("Receiver arg ({}) for {} must be the parameterStates[{}] "
                                          "of a MappingProjection (rather than a {})".
                                          format(self.receiver,
                                                 self.name,
                                                 MATRIX,
                                                 self.mappingProjection.__class__.__name__))
            if not isinstance(self.receiver.function.__self__, LinearCombination):
                raise LearningProjectionError("Function of receiver arg ({}) for {} must be a {} (rather than {})".
                                          format(self.receiver,
                                                 self.name,
                                                 LINEAR_COMBINATION_FUNCTION,
                                                 self.mappingProjection.function.__self__.__class__.__name__))

            # # MODIFIED 10/29/16 NEW:
            # # Reciever must be the parameterState for a MappingProjection with a LinearCombination identity function
            # if not isinstance(self.mappingProjection, MappingProjection):
            #     raise LearningProjectionError("Receiver arg ({}) for {} must be the parameterStates[{}] "
            #                               "of a MappingProjection (rather than a {})".
            #                               format(self.receiver,
            #                                      self.name,
            #                                      MATRIX,
            #                                      self.mappingProjection.__class__.__name__))
            # if not isinstance(self.receiver.function.__self__, LinearCombination):
            #     raise LearningProjectionError("Function of receiver arg ({}) for {} must be a {} (rather than {})".
            #                               format(self.receiver,
            #                                      self.name,
            #                                      LINEAR_FUNCTION,
            #                                      self.mappingProjection.function.__self__.__class__.__name__))
            # # MODIFIED 10/29/16 END


            # receiver is parameterState[MATRIX], so update its params with ones specified by LearningProjection
            # (by default, change LinearCombination.operation to SUM paramModulationOperation to ADD)
            if (self.mappingProjection.parameterStates and
                    self.receiver is self.mappingProjection.parameterStates[MATRIX]):
                self.receiver.paramsCurrent.update(weight_change_params)

            else:
                raise LearningProjectionError("Receiver arg ({}) for {} must be the "
                                          "parameterStates[{}] param of the receiver".
                                          format(self.receiver, self.name, MATRIX))

        # Receiver was specified as a MappingProjection
        elif isinstance(self.receiver, MappingProjection):

            self.mappingProjection = self.receiver

            from PsyNeuLink.Components.States.InputState import _instantiate_state_list
            from PsyNeuLink.Components.States.InputState import _instantiate_state

            # Check if MappingProjection has parameterStates Ordered Dict and MATRIX entry
            try:
                self.receiver.parameterStates[MATRIX]

            # receiver does NOT have parameterStates attrib
            except AttributeError:
                # Instantiate parameterStates Ordered dict
                #     with ParameterState for receiver's functionParams[MATRIX] param
                self.receiver.parameterStates = _instantiate_state_list(owner=self.receiver,
                                                                       state_list=[(MATRIX,
                                                                                    weight_change_params)],
                                                                       state_type=ParameterState,
                                                                       state_param_identifier=PARAMETER_STATE,
                                                                       constraint_value=self.mappingWeightMatrix,
                                                                       constraint_value_name=LEARNING_PROJECTION,
                                                                       context=context)
                self.receiver = self.receiver.parameterStates[MATRIX]

            # receiver has parameterStates but not (yet!) one for MATRIX, so instantiate it
            except KeyError:
                # Instantiate ParameterState for MATRIX
                self.receiver.parameterStates[MATRIX] = _instantiate_state(owner=self.receiver,
                                                                            state_type=ParameterState,
                                                                            state_name=MATRIX,
                                                                            state_spec=PARAMETER_STATE,
                                                                            state_params=weight_change_params,
                                                                            constraint_value=self.mappingWeightMatrix,
                                                                            constraint_value_name=LEARNING_PROJECTION,
                                                                            context=context)

            # receiver has parameterState for MATRIX, so update its params with ones specified by LearningProjection
            else:
                # MODIFIED 8/13/16:
                # FIX: ?? SHOULD THIS USE _assign_defaults:
                self.receiver.parameterStates[MATRIX].paramsCurrent.update(weight_change_params)

            # Assign self.receiver to parameterState used for weight matrix param
            self.receiver = self.receiver.parameterStates[MATRIX]

        # If it is not a ParameterState or MappingProjection, raise exception
        else:
            raise LearningProjectionError("Receiver arg ({}) for {} must be a MappingProjection or"
                                      " a MechanismParatemerState of one".format(self.receiver, self.name))

        if kwDeferredDefaultName in self.name:
            self.name = self.mappingProjection.name + ' ' + self.componentName
            # self.name = self.mappingProjection.name + \
            #             self.mappingProjection.parameterStates[MATRIX].name + \
            #             ' ' + self.componentName

        # Assign errorSource as the MappingProjection's receiver mechanism
        self.errorSource = self.mappingProjection.receiver.owner

        # GET RECEIVER'S WEIGHT MATRIX
        self._get_mapping_projection_weight_matrix()

        # Format input to MappingProjection's weight matrix
        # MODIFIED 8/19/16:
        # self.input_to_weight_matrix = np.zeros_like(self.mappingWeightMatrix[0])
        self.input_to_weight_matrix = np.zeros_like(self.mappingWeightMatrix.T[0])

        # Format output of MappingProjection's weight matrix
        # Note: this is used as a template for output value of its receiver mechanism (i.e., to which it projects)
        #       but that may not yet have been instantiated;  assumes that format of input = output for receiver mech
        # MODIFIED 8/19/16:
        # self.output_of_weight_matrix = np.zeros_like(self.mappingWeightMatrix.T[0])
        self.output_of_weight_matrix = np.zeros_like(self.mappingWeightMatrix[0])

    def _get_mapping_projection_weight_matrix(self):
        """Get weight matrix for MappingProjection to which LearningProjection projects

        """

        message = "PROGRAM ERROR: {} has either no {} or no {} param in paramsCurent".format(self.receiver.name,
                                                                                             FUNCTION_PARAMS,
                                                                                             MATRIX)
        if isinstance(self.receiver, ParameterState):
            try:
                self.mappingWeightMatrix = self.mappingProjection.matrix
            except KeyError:
                raise LearningProjection(message)

        elif isinstance(self.receiver, MappingProjection):
            try:
                self.mappingWeightMatrix = self.receiver.matrix
            except KeyError:
                raise LearningProjection(message)

    def _instantiate_sender(self, context=None):
        # DOCUMENT: SEE UPDATE BELOW
        """Assign self.variable to MonitoringMechanism output or self.receiver.receivererror_signals

        Call this after _instantiate_receiver, as that is needed to determine the sender (i.e., source of error_signal)

        If sender arg or PROJECTION_SENDER was specified, it has been assigned to self.sender
            and has been validated as a MonitoringMechanism, so:
            - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
            - assign its outputState.value as self.variable
        If sender was not specified (i.e., passed as MonitoringMechanism_Base specified in paramClassDefaults):
           if the owner of the MappingProjection projects to a MonitoringMechanism, then
               - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
               - assign its outputState.value as self.variable
           UPDATE: otherwise, if MappingProjection's receiver has an error_signal attribute, use that as self.variable
               (e.g., "hidden units in a multilayered neural network, using BackPropagation Function)
           [TBI: otherwise, implement default MonitoringMechanism]
           otherwise, raise exception

FROM TODO:
#    - _instantiate_sender:
#        - examine mechanism to which MappingProjection projects:  self.receiver.owner.receiver.owner
#            - check if it is a terminal mechanism in the system:
#                - if so, assign:
#                    - ComparatorMechanism MonitoringMechanism
#                        - ProcessInputState for ComparatorMechanism (name it??) with projection to target inputState
#                        - MappingProjection from terminal ProcessingMechanism to LinearCompator sample inputState
#                - if not, assign:
#                    - WeightedSum MonitoringMechanism
#                        - MappingProjection from preceding MonitoringMechanism:
#                            preceding processing mechanism (ppm):
#                                ppm = self.receiver.owner.receiver.owner
#                            preceding processing mechanism's output projection (pop)
#                                pop = ppm.outputState.projections[0]
#                            preceding processing mechanism's output projection learning signal (popls):
#                                popls = pop.parameterState.receivesFromProjections[0]
#                            preceding MonitoringMechanism (pem):
#                                pem = popls.sender.owner
#                            assign MappingProjection from pem.outputState to self.inputState
#                        - Get weight matrix for pop (pwm):
#                                pwm = pop.parameterState.params[MATRIX]

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
# IMPLEMENT: rename .monitoringMechanism -> .objective_mechanism
#            rename .errorSource -> .error_source

        # FIX: 8/7/16
        # FIX: NEED TO DEAL HERE WITH CLASS SPECIFICATION OF MonitoringMechanism AS DEFAULT
        # FIX: OR HAVE ALREADY INSTANTIATED DEFAULT MONITORING MECHANISM BEFORE REACHING HERE
        # FIX: EMULATE HANDLING OF DefaultMechanism (for MappingProjection) AND DefaultController (for ControlProjection)

        # FIX: 8/18/16
        # FIX: ****************
        # FIX: ASSIGN monitoring_source IN ifS, NOT JUST else
        # FIX: SAME FOR self.errorSource??

        objective_mechanism = None

        # ObjectiveMechanism specified for sender, so re-assign to its outputState
        if isinstance(self.sender, ObjectiveMechanism):
            self.sender = self.sender.outputState

        # OutputState specified (or re-assigned) for sender
        if isinstance(self.sender, OutputState):
            # Validate that it belongs to an ObjectiveMechanism being used for learning
            if not _objective_mechanism_role(self.sender.owner, LEARNING):
                raise LearningProjectionError("OutputState ({}) specified as sender for {} belongs to a {}"
                                          " rather than an ObjectiveMechanism with role=LEARNING".
                                          format(self.sender.name,
                                                 self.name,
                                                 self.sender.owner.__class__.__name__))
            self._validate_error_signal(self.sender.value)

            # - assign ObjectiveMechanism's outputState.value as self.variable
            # FIX: THIS DOESN"T SEEM TO HAPPEN HERE.  DOES IT HAPPEN LATER??

            # Add reference to ObjectiveMechanism to MappingProjection
            objective_mechanism = self.sender

        # ObjectiveMechanism class (i.e., not an instantiated object) specified for sender, so instantiate it:
        # - for terminal mechanism of Process, instantiate with Comparator function
        # - for preceding mechanisms, instantiate with WeightedError function
        else:
            # Get errorSource:  ProcessingMechanism for which error is being monitored
            #    (i.e., the mechanism to which the MappingProjection projects)
            # Note: MappingProjection._instantiate_receiver has not yet been called, so need to do parse below
            from PsyNeuLink.Components.States.InputState import InputState
            if isinstance(self.mappingProjection.receiver, Mechanism):
                self.errorSource = self.mappingProjection.receiver
            elif isinstance(self.mappingProjection.receiver, InputState):
                self.errorSource = self.mappingProjection.receiver.owner

            next_level_objective_mech_output = None

            # Check if errorSource has a projection to an ObjectiveMechanism or some other type of ProcessingMechanism
            for projection in self.errorSource.outputState.sendsToProjections:
                # errorSource has a projection to an ObjectiveMechanism being used for learning,
                #  so validate it, assign it, and quit search
                if _objective_mechanism_role(projection.receiver.owner, LEARNING):
                    self._validate_error_signal(projection.receiver.owner.outputState.value)
                    objective_mechanism = projection.receiver.owner
                    break
                # errorSource has a projection to a ProcessingMechanism, so:
                #   - determine whether that has a LearningProjection
                #   - if so, get its MonitoringMechanism and weight matrix (needed by BackProp)
                if isinstance(projection.receiver.owner, ProcessingMechanism_Base):
                    try:
                        next_level_learning_projection = projection.parameterStates[MATRIX].receivesFromProjections[0]
                    except (AttributeError, KeyError):
                        # Next level's projection has no parameterStates, Matrix parameterState or projections to it
                        #    => no LearningProjection
                        pass # FIX: xxx ?? ADD LearningProjection here if requested?? or intercept error message to do so?
                    else:
                        # Next level's projection has a LearningProjection so get:
                        #     the weight matrix for the next level's projection
                        #     the MonitoringMechanism that provides error_signal
                        # next_level_weight_matrix = projection.matrix
                        next_level_objective_mech_output = next_level_learning_projection.sender

            # errorSource does not project to an ObjectiveMechanism used for learning
            if not objective_mechanism:

                # FIX:  NEED TO DEAL WITH THIS RE: RL -> DON'T CREATE BACK PROJECTIONS??
                # NON-TERMINAL Mechanism
                # errorSource at next level projects to a MonitoringMechanism:
                #    instantiate ObjectiveMechanism configured with WeightedError Function
                #    (computes contribution of each element in errorSource to error at level to which it projects)
                #    and the back-projection for its error signal:
                if next_level_objective_mech_output:
                    error_signal = np.zeros_like(next_level_objective_mech_output.value)
                    next_level_output = projection.receiver.owner.outputState
                    activity = np.zeros_like(next_level_output.value)
                    matrix=projection.parameterStates[MATRIX]
                    derivative = next_level_objective_mech_output.sendsToProjections[0].\
                        receiver.owner.receiver.owner.function_object.derivative
                    from PsyNeuLink.Components.Functions.Function import WeightedError
                    objective_mechanism = ObjectiveMechanism(monitored_values=[next_level_output,
                                                                       next_level_objective_mech_output],
                                                              names=['ACTIVITY','ERROR_SIGNAL'],
                                                              function=ErrorDerivative(variable_default=[activity,
                                                                                                       error_signal],
                                                                                       derivative=derivative),
                                                              role=LEARNING,
                                                              name=self.mappingProjection.name + " Error_Derivative")
                # TERMINAL Mechanism
                # errorSource at next level does NOT project to an ObjectiveMechanism:
                #     instantiate ObjectiveMechanism configured as a comparator
                #         that compares errorSource output with external training signal
                else:
                    # Instantiate ObjectiveMechanism to receive the (externally provided) target for training
                    try:
                        sample_state_name = self.errorSource.paramsCurrent[MONITOR_FOR_LEARNING]
                        sample_source = self.errorSource.outputStates[sample_state_name]
                        sample_size = np.zeros_like(sample_source)
                    except KeyError:
                        # No state specified so use Mechanism as sender arg
                        sample_source = self.errorSource
                        sample_size = np.zeros_like(self.errorSource.outputState.value)

                    # Assign output_signal to output of errorSource
                    if self.function.componentName is BACKPROPAGATION_FUNCTION:
                        target_size = np.zeros_like(self.errorSource.outputState.value)
                    # Force sample and target of Comparartor to be scalars for RL
                    elif self.function.componentName is RL_FUNCTION:
                        sample_size = np.array([0])
                        target_size = np.array([0])
                    else:
                        raise LearningProjectionError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                                  format(self.function.name, self.name))

                    # IMPLEMENTATION NOTE: specify target as a template value (matching the sample's output value)
                    #                      since its projection (from either a ProcessInputState or a SystemInputState)
                    #                      will be instantiated by the Composition object to which the mechanism belongs
                    # FIX: FOR RL, NEED TO BE ABLE TO CONFIGURE OBJECTIVE MECHANISM WITH SCALAR INPUTSTATES
                    # FIX:         AND FULL CONNECTIVITY MATRICES FROM THE MONITORED OUTPUTSTATES
                    objective_mechanism = ObjectiveMechanism(default_input_value=[sample_size, target_size],
                                                             monitored_values=[sample_source, target_size],
                                                             names=[SAMPLE,TARGET],
                                                             # FIX: WHY DO WEIGHTS HAVE TO BE AN ARRAY HERE??
                                                             function=LinearCombination(weights=[[-1], [1]]),
                                                             role=LEARNING,
                                                             params= {OUTPUT_STATES:
                                                                          [{NAME:TARGET_ERROR},
                                                                           {NAME:TARGET_ERROR_MEAN,
                                                                            CALCULATE:lambda x: np.mean(x)},
                                                                           {NAME:TARGET_ERROR_SUM,
                                                                            CALCULATE:lambda x: np.sum(x)},
                                                                           {NAME:TARGET_SSE,
                                                                            CALCULATE:lambda x: np.sum(x*x)},
                                                                           {NAME:TARGET_MSE,
                                                                            CALCULATE:lambda x: np.sum(x*x)/len(x)}]},
                                                             name=self.mappingProjection.name + " Target_Error")
                    objective_mechanism.learning_role = TARGET

            self.sender = objective_mechanism.outputState

            # "Cast" self.variable to match value of sender (MonitoringMechanism) to pass validation in add_to()
            # Note: self.variable will be re-assigned in _instantiate_function()
            self.variable = self.error_signal

            # Add self as outgoing projection from MonitoringMechanism
            from PsyNeuLink.Components.Projections.Projection import _add_projection_from
            _add_projection_from(sender=objective_mechanism,
                                state=objective_mechanism.outputState,
                                projection_spec=self,
                                receiver=self.receiver,
                                context=context)

        # VALIDATE THAT OUTPUT OF SENDER IS SAME LENGTH AS THIRD ITEM (ERROR SIGNAL) OF SEL.FFUNCTION.VARIABLE

        # Add reference to MonitoringMechanism to MappingProjection
        self.mappingProjection.monitoringMechanism = objective_mechanism

    def _validate_error_signal(self, error_signal):
        """Check that error signal (MonitoringMechanism.outputState.value) conforms to what is needed by self.function
        """

        if self.function.componentName is RL_FUNCTION:
            # The length of the sender (MonitoringMechanism)'s outputState.value (the error signal) must == 1
            #     (since error signal is a scalar for RL)
            if len(error_signal) != 1:
                raise LearningProjectionError("Length of error signal ({}) received by {} from {}"
                                          " must be 1 since {} uses {} as its learning function".
                                          format(len(error_signal), self.name, self.sender.owner.name, self.name, RL_FUNCTION))
        if self.function.componentName is BACKPROPAGATION_FUNCTION:
            # The length of the sender (MonitoringMechanism)'s outputState.value (the error signal) must be the
            #     same as the width (# columns) of the MappingProjection's weight matrix (# of receivers)
            if len(error_signal) != self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]:
                raise LearningProjectionError("Length of error signal ({}) received by {} from {} must match the"
                                          "receiver dimension ({}) of the weight matrix for {}".
                                          format(len(error_signal),
                                                 self.name,
                                                 self.sender.owner.name,
                                                 len(self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]),
                                                 self.mappingProjection))
        else:
            raise LearningProjectionError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                      format(self.function.name, self.name))

    def _instantiate_function(self, context=None):
        """Construct self.variable for input to function, call super to instantiate it, and validate output

        function implements function to compute weight change matrix for receiver (MappingProjection) from:
        - input: array of sender values (rows) to MappingProjection weight matrix (self.variable[0])
        - output: array of receiver values (cols) for MappingProjection weight matrix (self.variable[1])
        - error:  array of error signals for receiver values (self.variable[2])
        """

        # Reconstruct self.variable as input for function
        self.variable = [[0]] * 3
        self.variable[0] = self.input_to_weight_matrix
        self.variable[1] = self.output_of_weight_matrix
        self.variable[2] = self.error_signal

        super()._instantiate_function(context)

        from PsyNeuLink.Components.Functions.Function import ACTIVATION_FUNCTION, TransferFunction
        # Insure that the learning function is compatible with the activation function of the errorSource
        error_source_activation_function_type = type(self.errorSource.function_object)
        function_spec = self.function_object.paramsCurrent[ACTIVATION_FUNCTION]
        if isinstance(function_spec, TransferFunction):
            learning_function_activation_function_type = type(function_spec)
        elif issubclass(function_spec, TransferFunction):
            learning_function_activation_function_type = function_spec
        else:
            raise LearningProjectionError("PROGRAM ERROR: activation function ({}) for {} is not a TransferFunction".
                                      format(function_spec, self.name))
        if error_source_activation_function_type != learning_function_activation_function_type:
            raise LearningProjectionError("Activation function ({}) of error source ({}) is not compatible with "
                                      "the activation function ({}) specified for {}'s function ({}) ".
                                      format(error_source_activation_function_type.__name__,
                                             self.errorSource.name,
                                             learning_function_activation_function_type.__name__,
                                             self.name,
                                             self.params[FUNCTION].__self__.__class__.__name__))

        # FIX: MOVE TO AFTER INSTANTIATE FUNCTION??
        # IMPLEMENTATION NOTE:  MOVED FROM _instantiate_receiver
        # Insure that LearningProjection output (error signal) and receiver's weight matrix are same shape
        try:
            receiver_weight_matrix_shape = self.mappingWeightMatrix.shape
        except TypeError:
            # self.mappingWeightMatrix = 1
            receiver_weight_matrix_shape = 1
        try:
            LEARNING_PROJECTION_shape = self.value.shape
        except TypeError:
            LEARNING_PROJECTION_shape = 1

        if receiver_weight_matrix_shape != LEARNING_PROJECTION_shape:
            raise ProjectionError("Shape ({0}) of matrix for {1} learning signal from {2}"
                                  " must match shape of receiver weight matrix ({3}) for {4}".
                                  format(LEARNING_PROJECTION_shape,
                                         self.name,
                                         self.sender.name,
                                         receiver_weight_matrix_shape,
                                         # self.receiver.owner.name))
                                         self.mappingProjection.name))


    @property
    def error_signal(self):
        return self.sender.value

    @property
    def monitoringMechanism(self):
        return self.sender.owner
