# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningAuxiliary ******************************************************

"""
.. _LearningAuxiliary_Overview:

Overview
--------

This module contains functions that automatically construct and compose the components necessary for learning
when this is specified for a process.

COMMENT:
    IMPLEMENT: LearningMechanism:
        PROCESS & SYSTEM:
          • Convert ProcessInputPort and SystemInputPort into Mechanisms with LinearFunction IDENTITY_FUNCTION
          • Use only one ObjectiveMechanism for all levels with the following args:
                default_variable[[ACTIVITY][ERROR]]
                monitored_output_ports: [[error_mech.OutputPort][error_mech.objective_mechanism.OutputPort]]
                names: [[ACTIVITY][ERROR]]
                function:  ErrorDerivative(variable, derivative)
                               variable[0] = activity
                               variable[1] = error_signal from error_mech ObjectiveMechanism (target for TERMINAL)
                               derivative = error_derivative (1 for TERMINAL)
                role:LEARNING
          • Use only one LearningMechanism with the following args:
                variable[[ACTIVATION_INPUT_INDEX][ACTIVATION_OUTPUT_INDEX][ERROR_SIGNAL_INDEX]
                activation_derivative
                error_matrix
                function
            Initialize and assign args with the following WIZZARD:
        WIZZARD:
            Needs to know
                activation_mech_output (Mechanism)
                    activation_derivative (function)
                error_mech (Mechanism)
                    error_derivative (function)
                    error_matrix (ndarray) - for MappingProjection from activation_mech_output to error_mech
            ObjectiveMechanism:
                Initialize variable:
                      use error_mech.outputPort.valuee to initialize variable[ACTIVITY]
                      use outputPort.value of error_mech's objective_mechanism to initialize variable[ERROR]
                Assign mapping projections:
                      nextLevel.outputPort.value -> input_ports[ACTIVITY] of ObjectiveMechanism
                      nextLevel.objective_mechanism.outputPort.value  -> input_ports[ERROR] of ObjectiveMechanism
                NOTE: For `TERMINAL` Mechanism:
                          error_mech is Process or System InputPort (function=Linear, so derivative =1), so that
                             error_mech.outputPort.value is the target, and
                             error_derivative = 1
                             error_matrix = IDENTITY_MATRIX (this should be imposed)
            LearningMechanism:
                Initialize variable:
                      use mapping_projection.sender.value to initialize variable[ACTIVATION_INPUT_INDEX]
                      use activation_mech_output_outputPort.value to initialize variable[ACTIVATION_OUTPUT_INDEX]
                      use error_mech.objecdtive_mechanism.OutputPort.value to initialize variable[ERROR_SIGNAL_INDEX]
                Assign activation_derivative using function of activation_mech_output of mapping_projection (one being learned)
                Assign error_derivative using function of error_mech
                Assign error_matrix as runtime_param using projection to error_mech [ALT: ADD TO VARIABLE]
                Assign mapping projections:
                      mapping_projection.sender -> input_ports[ACTIVATION_INPUT_INDEX] of LearningMechanism
                      activation_mech_output.outputPort -> input_ports[ACTIVATION_OUTPUT_INDEX] of LearningMechanism
                      error_mech.objective_mechanism.OutputPort.value -> input_ports[ERROR_SIGNAL_INDEX]

            For TARGET MECHANISM:  Matrix is IDENTITY MATRIX??
            For TARGET MECHANISM:  derivative for ObjectiveMechanism IDENTITY FUNCTION


    *************************************************************************

    Call in _instantiate_attributes_before_function() of LearningProjection

    Do the following:
        Get:
            activation_mech_projection (one being learned) (MappingProjection)
            activation_output_mech (ProcessingMechanism)
            activation_mech_input (OutputPort)
            activation_mech_output (OutputPort)
            activation_derivative (function)
            error_matrix (ParameterPort)
            error_derivative (function)
            error_mech (error_source_mech) (ProcessingMechanism)
            error_signal_mech (LearningMechanism or ObjectiveMechanism)
            error_signal_mech_output (OutputPort)
            error_objective_mech (error_source_objective_mech) (ObjectiveMechanism)

        Instantiate:
            ObjectiveMechanism:
                Construct with:
                   monitor_values: [error_mech.outputPort.value, error_mech.objective_mech.outputPort.value]
                   names = [ACTIVITY/SAMPLE, ERROR/TARGET]
                   function = ErrorDerivative(derivative=error_derivative)
                NOTE: For `TERMINAL` Mechanism:
                          error_mech is Process or System InputPort (function=Linear, so derivative =1), so that
                             error_mech.outputPort.value is the target, and
                             error_derivative = 1
                             error_matrix = IDENTITY_MATRIX (this should be imposed)
            LearningMechanism:
                Construct with:
                    variable=[[activation_mech_input],[activation_mech_output],[ObjectiveMechanism.outputPort.value]]
                    error_matrix = error_matrix
                    function = one specified with Learning specification
                               NOTE: can no longer be specified as function for LearningProjection
                    names = [ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL]
                        NOTE:  this needs to be implemented for LearningMechanism as it is for ObjectiveMechanism
                Check that, if learning function expects a derivative, that the one specified
                    is compatible with the function of the activation_output_mech
                Assign:
                    NOTE:  should do these in their own Learning module function, called by LearningMechanaism directly
                        as is done for ObjectiveMechanism
                    MappingProjection: activation_mech_projection.sender -> LearningMechanism.input_port[ACTIVATION_INPUT]
                    MappingProjection: activation_output_mech.outputPort -> LearningMechanism.input_port[ACTIVATION_OUTPUT]
                    MappingProjection: ObjectiveMechanism -> LearningMechanism.input_port[ERROR_SIGNAL]


COMMENT


.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

import types
import warnings

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.combinationfunctions import PredictionErrorDeltaFunction
from psyneulink.core.components.functions.learningfunctions import BackPropagation, Hebbian, Reinforcement, TDLearning
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import ACTIVATION_INPUT, \
    ACTIVATION_OUTPUT, ERROR_SIGNAL, LearningMechanism, LearningType, LearningTiming
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.shellclasses import Function
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.keywords import \
    BACKPROPAGATION_FUNCTION, COMPARATOR_MECHANISM, HEBBIAN_FUNCTION, IDENTITY_MATRIX, LEARNING, LEARNING_MECHANISM, \
    MATRIX, MONITOR_FOR_LEARNING, NAME, OUTCOME, PREDICTION_ERROR_MECHANISM, PROJECTIONS, RL_FUNCTION, SAMPLE, \
    TARGET, TDLEARNING_FUNCTION, VARIABLE, WEIGHT
from psyneulink.library.components.mechanisms.processing.objective.predictionerrormechanism import PredictionErrorMechanism

__all__ = [
    'LearningAuxiliaryError'
]


class LearningAuxiliaryError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def _instantiate_learning_components(learning_projection, context=None):
    """Instantiate learning components for a LearningProjection

    Instantiates a LearningMechanism or ObjectiveMechanism as the sender for each learning_projection in a learning
        sequence.  A learning sequence is defined as a sequence of ProcessingMechanisms, each of which has a
        projection — that has been specified for learning — to the next Mechanism in the sequence.  This method
        instantiates the components required to support learning for those projections (most importantly,
        the LearningMechanism that provide them with the learning_signal required to modify the matrix of the
        projection, and the ObjectiveMechanism that calculates the error_signal used to generate the learning_signals).


    It instantiates a LearningMechanism or ObjectiveMechanism as the sender for the learning_projection:
    - a LearningMechanism for projections to any ProcessingMechanism that is not the last in the learning sequence;
    - an ObjectiveMechanism for projections to a ProcessingMechanism that is the last in the learning sequence

    Assume that learning_projection's variable and parameters have been specified and validated,
       (which is the case when this method is called from the learning_projection itself in _instantiate_sender()).

    Notes:

    * Once the `receiver` for the learning_projection has been identified, or instantiated:
        - it is thereafter referred to (by reference to it owner) as the `activation_mech_projection`,
            and the Mechanism to which it projects as the `activation_output_mech`;
        - the Mechanism to which it projects is referred referred to as the error_mech (the source of the error_signal).

    * See LearningComponents class for the names of the components of learning used here.

    * This method supports only a single pathway for learning;  that is, the learning sequence must be a linear
        sequence of ProcessingMechanisms.  This is consistent with its implementation at the Process level;
        convergent and divergent pathways for learning can be accomplished through Composition in a
        System.  Accordingly:

            - each LearningMechanism can have only one LearningProjection
            - each ProcessingMechanism can have only one MappingProjection that is subject to learning

      When searching downstream for projections that are being learned (to identify the error_mech Mechanism as a
      source for the LearningMechanism's error_signal), the method uses the first projection that it finds, beginning
      in `primary outputPort <OutputPort_Primary>` of the activation_mech_output, and continues similarly
      in the error_mech.  If any conflicting implementations of learning components are encountered,
      an exception is raised.

    """
    # IMPLEMENTATION NOTE: CURRENT IMPLEMENTATION ONLY SUPPORTS CALL FROM LearningProjection._instantiate_sender();
    #                      IMPLEMENTATION OF SUPPORT FOR EXTERNAL CALLS:
    #                      - SHOULD CHECK UP FRONT WHETHER SENDER OR RECEIVER IS SPECIFIED, AND BRANCH ACCORDINGLY
    #                            FAILURE TO SPECIFY BOTH SHOULD RAISE AN EXCEPTION (OR FURTHER DEFER INSTANTIATION?)
    #                      - WILL REQUIRE MORE EXTENSIVE CHECKING AND VALIDATION
    #                              (E.G., OF WHETHER ANY LearningMechanism IDENTIFIED HAVE A PROJECTION FROM AN
    #                               APPROPRIATE ObjectiveMechanism, etc.
    from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import \
        ComparatorMechanism, MSE

    # Call should generally be from LearningProjection._instantiate_sender,
    #    but may be used more generally in the future
    if context.source != ContextFlags.METHOD:
        raise LearningAuxiliaryError("PROGRAM ERROR".format("_instantiate_learning_components only supports "
                                                             "calls from a LearningProjection._instantiate_sender()"))


    # First determine whether receiver for LearningProjection has been instantiated and, if not, instantiate it:
    #    this is required to instantiate the LearningMechanism, as well as an ObjectiveMechanism if necessary.
    try:
        isinstance(learning_projection.receiver.owner, MappingProjection)
    except AttributeError:
        _instantiate_receiver_for_learning_projection(learning_projection, context=context)
        # IMPLEMENTATION NOTE: THIS SHOULD IDENTIFY/CONFIRM RECEIVER BASED ON PROJECTIONS TO THE OBJECTIVE MECHANISM
        #                      OF THE learning_projection's LearningMechanism;  IF THERE ARE NONE, THEN RAISE EXCEPTION

    # Next, validate that the receiver does not receive any other LearningProjections
    # IMPLEMENTATION NOTE:  this may be overly restrictive in the context of a system --
    #                       will need to be dealt with in Composition implementation (by examining its learning graph??)
    if any((isinstance(projection, LearningProjection) and not projection is learning_projection) for projection in
           learning_projection.receiver.mod_afferents):
        raise LearningAuxiliaryError("{} can't be assigned as LearningProjection to {} since that already has one.".
                                     format(learning_projection.name, learning_projection.receiver.owner.name))

    # Now that the receiver has been identified, use it to get the components (lc) needed for learning.
    # Note: error-related components may not yet be defined (if LearningProjection is for a TERMINAL mechanism)
    lc = LearningComponents(learning_projection=learning_projection)

    # Check if activation_output_mech already has a projection to an ObjectiveMechanism or a LearningMechanism
    #    (i.e., the source of an error_signal for learning_projection)
    # IMPLEMENTATION NOTE: this uses the first projection found (starting with the primary outputPort)

    # FIX: CHECK THAT THE PROJECTION IS TO ANOTHER MECHANISM IN THE CURRENT PROCESS;
    #      OTHERWISE, ALLOW OBJECTIVE_MECHANISM TO BE IMPLEMENTED


    # Check whether activation_output_mech belongs to more than one process and, if it does, it is the TERMINAL
    #    the one currently being instantiated, in which case it needs to be assigned an ObjectiveMechanism
    #
    # If the only MappingProjections from activation_output_mech are to mechanisms that are not in the same
    #    process as the mechanism that projects to it (i.e., lc.activation_mech_input.owner), then:
    #    - set is_target to True so that it will be assigned an ObjectiveMechanism
    # Note: this handles the case in which the current mech belongs to more than one process, and is the
    #    the TERMINAL of the one currently being instantiated.
    # IMPLEMENTATION NOTE:  could check whether current mech is a TERMINAL for the process currently being
    #                       instantiated, identified by the mechanism that projects to it.  However, the latter
    #                       may also belong to more than one process, so would have to sort that out.  Current
    #                       implementation doesn't have to worry about that.

    objective_mechanism = None

    # if activation_output_mech has outgoing projections
    if (lc.activation_mech_output.efferents and
            # if the ProcessingMechanisms to which activation_mech_output projects do not belong to any of the same
            # processes to which the mechanisms that project to activation_output_mech belong, then it should be
            # treated as a TERMINAL and is_target should be set to True
            not any(
                        isinstance(projection.receiver.owner, ProcessingMechanism_Base) and
                        any(        # processes of ProcessingMechanisms to which activation_output_mech projects
                                    process in projection.receiver.owner.processes
                                    # processes of mechanism that project to activation_output_mech
                                    for process in lc.activation_mech_input.owner.processes)
                        for projection in lc.activation_mech_output.efferents)):
        is_target = True

    else:

        for projection in lc.activation_mech_output.efferents:

            receiver_mech = projection.receiver.owner
            receiver_state = projection.receiver

            # Check if projection already projects to a LearningMechanism or an ObjectiveMechanism

            # activation_output_mech projects to a LearningMechanism
            if isinstance(receiver_mech, LearningMechanism):

                # IMPLEMENTATION NOTE:  THIS IS A SANITY CHECK;  IF THE learning_projection ALREADY HAS A SENDER
                #                       THAT IS A LearningMechanism, THIS FUNCTION SHOULD NOT HAVE BEEN CALLED
                # If receiver_mech is a LearningMechanism that is the sender for the learning_projection,
                #    raise an exception since this function should not have been called.
                if learning_projection.sender is receiver_mech:
                    raise LearningAuxiliaryError("PROGRAM ERROR: "
                                                  "{} already has a LearningMechanism as its sender ({})".
                                                 format(learning_projection.name, receiver_mech.name))

                # If receiver_mech for activation_output_mech is a LearningMechanism that receives projections to its:
                #     - ACTIVATION_INPUT InputPort from activation_mech_input.owner
                #     - ACTIVATION_OUTPUT InputPort from activation_output_mech
                #         (i.e., the mechanism before activation_output_mech in the learning sequence)
                #         then this should be the LearningMechanism for the learning_projection,
                #         so issue warning, assign it as the sender, and return
                if (receiver_state.name is ACTIVATION_OUTPUT and
                        any(projection.sender.owner is lc.activation_mech_input.owner
                            for projection in receiver_mech.input_ports[ACTIVATION_INPUT].path_afferents)):
                        warnings.warn("An existing LearningMechanism ({}) was found for and is being assigned to {}".
                                      format(receiver_mech.name, learning_projection.name))
                        learning_projection.sender = receiver_mech
                        return

            # activation_output_mech already projects to an ObjectiveMechanism used for learning
            #    (presumably instantiated for another process);
            #    note:  doesn't matter if it is not being used for learning (then its just another ProcessingMechanism)
            elif isinstance(receiver_mech, ObjectiveMechanism) and LEARNING in receiver_mech._role:

                # ObjectiveMechanism is for learning but projection is not to its SAMPLE inputPort
                if LEARNING in receiver_mech._role and not receiver_state.name is SAMPLE:
                    raise LearningAuxiliaryError("PROGRAM ERROR: {} projects to the {} rather than the {} "
                                                  "inputPort of an ObjectiveMechanism for learning {}".
                                                 format(lc.activation_output_mech.name,
                                                         receiver_state.name,
                                                         SAMPLE,
                                                         receiver_mech.name))

                # IMPLEMENTATION NOTE:  THIS IS A SANITY CHECK;  IF THE learning_projection ALREADY HAS A SENDER
                #                       THAT IS A LearningMechanism, THIS FUNCTION SHOULD NOT HAVE BEEN CALLED
                # If the ObjectiveMechanism projects to a LearningMechanism that is the sender for the
                #     learning_projection, raise exception as this function should not have been called
                elif (isinstance(learning_projection.sender, LearningMechanism) and
                          any(learning_projection.sender.owner is projection.receiver.owner
                              for projection in receiver_mech.output_port.efferents)):
                    raise LearningAuxiliaryError("PROGRAM ERROR:  {} already has an "
                                                  "ObjectiveMechanism ({}) and a "
                                                  "LearningMechanism ({}) assigned to it".
                                                 format(learning_projection.name,
                                                         receiver_mech.name,
                                                         learning_projection.sender.owner.name))

                else:
                    objective_mechanism = receiver_mech


        # FIX: CHECK THAT THE PROJECTION IS TO ANOTHER MECHANISM IN THE CURRENT PROCESS;
        #      OTHERWISE, ALLOW OBJECTIVE_MECHANISM TO BE IMPLEMENTED
        #      BUT HOW CAN IT KNOW THIS, SINCE UNTIL THE METHOD IS PART OF A COMPOSITION, IT DOESN'T KNOW WHAT
        #          COMPOSITION IS BEING CALLED.  -- LOOK AT LearningProjection_OLD TO SEE HOW IT WAS HANDLED THERE.
        #          OR IS A TARGET/OBJECTIVE MECHANISM ALWAYS ASSIGNED FOR A PROCESS, AND THEN DELETED IN SYSTEM GRAPH?
        #          IN THAT CASE, JUST IGNORE SECOND CONDITION BELOW (THAT IT HAS NO PROJECTIONS TO LEARNING_PROJECTIONS?

        # Next, determine whether an ObjectiveMechanism or LearningMechanism should be assigned as the sender
        # It SHOULD be an ObjectiveMechanism (i.e., TARGET) if either:
        #     - it has no outgoing projections or
        #     - it has no projections that receive a LearningProjection;
        #   in either case, lc.error_projection returns None.
        #   Note: this assumes that LearningProjections are being assigned from the end of the pathway to the beginning.
        is_target = not lc.error_projection

    # INSTANTIATE learning function

    # Note: have to wait to do it here, as Backpropagation needs error_matrix,
    #       which depends on projection to ObjectiveMechanism

    # IMPLEMENTATION NOTE:
    #      THESE SHOULD BE MOVE TO COMPOSITION WHEN IT IS IPMLEMENTED
    #      THESE SHOULD BE MOVED (ALONG WITH THE SPECIFICATION FOR LEARNING) TO A DEDICATED LEARNING SPEC
    #      FOR PROCESS AND SYSTEM, RATHER THAN USING A LearningProjection
    # Get function used for learning and the learning_rate from their specification in the LearningProjection
    # FIXME: learning_function is deprecated
    learning_function = learning_projection._init_args['learning_function']
    learning_rate = learning_projection._init_args['learning_rate']
    error_function = learning_projection._init_args['error_function']

    # HEBBIAN LEARNING FUNCTION
    if learning_function.componentName is HEBBIAN_FUNCTION:

        activation = np.zeros_like(lc.activation_mech_input.value)
        learning_rate = learning_projection.learning_function.learning_rate
        activation_output = error_signal = None

        # FIX: GET AND PASS ANY PARAMS ASSIGNED IN LearningProjection.learning_function ARG:
        # FIX:     ACTIVATION FUNCTION AND/OR LEARNING RATE
        learning_function = Hebbian(default_variable=activation,
                                    # activation_function=lc.activation_mech_fct,
                                    learning_rate=learning_rate)

        objective_mechanism = lc.activation_output_mech

    # REINFORCEMENT LEARNING FUNCTION
    elif learning_function.componentName is RL_FUNCTION:

        activation_input = np.zeros_like(lc.activation_mech_input.value)
        activation_output = np.zeros_like(lc.activation_mech_output.value)

        # Force output activity and error arrays to be scalars
        error_signal = np.array([0])
        error_output = np.array([0])
        learning_rate = learning_function.learning_rate

        # FIX: GET AND PASS ANY PARAMS ASSIGNED IN LearningProjection.learning_function ARG:
        # FIX:     ACTIVATION FUNCTION AND/OR LEARNING RATE
        learning_function = Reinforcement(default_variable=[activation_input, activation_output, error_signal],
                                          # activation_function=lc.activation_mech_fct,
                                          learning_rate=learning_rate)

    elif learning_function.componentName is TDLEARNING_FUNCTION:
        activation_input = np.zeros_like(lc.activation_mech_input.value)
        activation_output = np.zeros_like(lc.activation_mech_output.value)

        error_output = np.zeros_like(lc.activation_mech_output.value)
        error_signal = np.zeros_like(lc.activation_mech_output.value)
        learning_rate = learning_function.learning_rate

        learning_function = TDLearning(default_variable=[activation_input,
                                                         activation_output,
                                                         error_signal],
                                       # activation_function=lc.activation_mech_fct,
                                       learning_rate=learning_rate)

    # BACKPROPAGATION LEARNING FUNCTION
    elif learning_function.componentName is BACKPROPAGATION_FUNCTION:

        # Get activation_output_mech values
        activation_input = np.zeros_like(lc.activation_mech_input.value)
        activation_output = np.zeros_like(lc.activation_mech_output.value)

        # Validate that the function for activation_output_mech has a derivative
        try:
            activation_derivative = lc.activation_mech_fct.derivative
        except AttributeError:
            raise LearningAuxiliaryError("Function for activation_output_mech of {} "
                                          "must have a derivative to be used "
                                          "with {}".format(learning_projection.name,
                                                           BackPropagation.componentName))

        # Get error_mech values
        if is_target:
            error_output = np.ones_like(lc.activation_mech_output.value)
            error_signal = np.zeros_like(lc.activation_mech_output.value)
            error_matrix = np.identity(len(error_signal))
            # IMPLEMENTATION NOTE: Assign error_derivative to derivative of ProcessingInputPort or SystemInputPort
            #                      function when these are fully implemented as mechanisms
            # activation_derivative = Linear().derivative
            error_derivative = Linear().derivative

        else:
            error_output = np.zeros_like(lc.error_mech_output.value)
            error_signal = np.zeros_like(lc.error_signal_mech.output_ports[ERROR_SIGNAL].value)
            error_matrix = lc.error_matrix
            try:
                error_derivative = lc.error_derivative
            except AttributeError:
                raise LearningAuxiliaryError("Function for error_mech of {} "
                                              "must have a derivative to be "
                                              "used with {}".format(learning_projection.name,
                                                                    BackPropagation.componentName))

        # FIX: GET AND PASS ANY PARAMS ASSIGNED IN LearningProjection.learning_function ARG:
        # FIX:     DERIVATIVE, LEARNING_RATE, ERROR_MATRIX
        learning_function = BackPropagation(default_variable=[activation_input,
                                                              activation_output,
                                                              # error_output,
                                                              error_signal],
                                            activation_derivative_fct=activation_derivative,
                                            learning_rate=learning_rate)

    else:
        raise LearningAuxiliaryError("PROGRAM ERROR: unrecognized learning "
                                      "function ({}) for {}".format(learning_function.componentName,
                                                                    learning_projection.name))


    # INSTANTIATE ObjectiveMechanism

    # If it is a TARGET, instantiate an ObjectiveMechanism and assign as learning_projection's sender
    if is_target:

        if objective_mechanism is None:
            # Instantiate ObjectiveMechanism
            # Notes:
            # * MappingProjections for ObjectiveMechanism's input_ports will be assigned in its own call to Composition
            # * Need to specify both default_variable and monitored_output_ports since they may not be the same
            #    sizes (e.g., for RL the monitored_output_port for the sample may be a vector, but its input_value must be scalar)
            # SAMPLE inputPort for ObjectiveMechanism should come from activation_mech_output
            # TARGET inputPort for ObjectiveMechanism should be specified by string (TARGET),
            #     so that it is left free to later be assigned a projection from ProcessInputPort and/or SystemInputPort
            # Assign derivative of Linear to lc.error_derivative (as default, until TARGET projection is assigned);
            #    this will induce a simple subtraction of target-sample (i.e., implement a comparator)
            sample_input = target_input = error_output
            # MODIFIED 10/10/17 OLD:
            # objective_mechanism = ComparatorMechanism(sample=lc.activation_mech_output,
            #                                           target=TARGET,
            #                                           # input_ports=[sample_input, target_input],
            #                                           # FOR TESTING: ALTERNATIVE specifications of input_ports arg:
            #                                           # input_ports=[(sample_input, FULL_CONNECTIVITY_MATRIX),
            #                                           #               target_input],
            #                                           # input_ports=[(sample_input, RANDOM_CONNECTIVITY_MATRIX),
            #                                           #               target_input],
            #                                           input_ports=[{NAME:SAMPLE,
            #                                                          VARIABLE:sample_input,
            #                                                          WEIGHT:-1
            #                                                          },
            #                                                         {NAME:TARGET,
            #                                                          VARIABLE:target_input,
            #                                                          # WEIGHT:1
            #                                                          }],
            #                                           name="{} {}".format(lc.activation_output_mech.name,
            #                                                               COMPARATOR_MECHANISM))
            # MODIFIED 10/10/17 NEW:
            if learning_function.componentName == TDLEARNING_FUNCTION:
                objective_mechanism = PredictionErrorMechanism(
                        sample={NAME: SAMPLE,
                                VARIABLE: sample_input,
                                PROJECTIONS: [lc.activation_mech_output]},
                        target={NAME: TARGET,
                                VARIABLE: target_input},
                        function=PredictionErrorDeltaFunction(gamma=1.0),
                        name="{} {}".format(lc.activation_output_mech.name,
                                            PREDICTION_ERROR_MECHANISM))
            else:
                objective_mechanism = ComparatorMechanism(sample={NAME: SAMPLE,
                                                                  VARIABLE: sample_input,
                                                                  PROJECTIONS: [lc.activation_mech_output],
                                                                  WEIGHT: -1},
                                                          target={NAME: TARGET,
                                                                  VARIABLE: target_input},
                                                          function=error_function,
                                                          output_ports=[OUTCOME, MSE],
                                                          name="{} {}".format(lc.activation_output_mech.name,
                                                                              COMPARATOR_MECHANISM),
                                                          context=context)
                # MODIFIED 10/10/17 END

            # # FOR TESTING: ALTERNATIVE to Direct call to ObjectiveMechanism
            # #              (should produce identical result to use of ComparatorMechanism above)
            # objective_mechanism = ObjectiveMechanism(monitored_output_ports=[lc.activation_mech_output,
            #                                                            TARGET],
            #                                          input_ports=[{SAMPLE:sample_input},
            #                                                        {TARGET:target_input}],
            #                                          function=LinearCombination(weights=[[-1], [1]]),
            #                                          output_ports=[ERROR_SIGNAL,
            #                                                         {NAME:MSE,
            #                                                          ASSIGN:lambda x: np.sum(x*x)/len(x)}],
            #                                          name="\'{}\' {}".format(lc.activation_output_mech.name,
            #                                                                  COMPARATOR_MECHANISM))

            objective_mechanism._role = LEARNING
            objective_mechanism._learning_role = TARGET

        try:
            lc.error_projection = objective_mechanism.input_port.path_afferents[0]
            # FIX: THIS IS TO FORCE ASSIGNMENT (SINCE IT DOESN'T SEEM TO BE
            # ASSIGNED BY TEST BELOW)
        except AttributeError:
            raise LearningAuxiliaryError("PROGRAM ERROR: problem finding "
                                          "projection to TARGET "
                                          "ObjectiveMechanism from {} when "
                                          "instantiating {}".format(
                lc.activation_output_mech.name,
                learning_projection.name))
        else:
            if not lc.error_matrix:
                raise LearningAuxiliaryError("PROGRAM ERROR: problem "
                                              "assigning error_matrix for "
                                              "projection to "
                                              "ObjectiveMechanism for {} when "
                                              "instantiating {}".format(
                    lc.activation_output_mech.name,
                    learning_projection.name))

        # INSTANTIATE LearningMechanism

    # - LearningMechanism incoming projections (by inputPort):
    #    ACTIVATION_INPUT:
    #        MappingProjection from activation_mech_input
    #    ACTIVATION_OUTPUT:
    #        MappingProjection from activation_mech_output
    #    ERROR_SIGNAL:
    #        specified in error_source argument:
    #        Note:  the error_source for LearningMechanism is set in lc.error_signal_mech:
    #            if is_target, this comes from the primary outputPort of objective_mechanism;
    #            otherwise, it comes from output_ports[ERROR_SIGNAL] of the LearningMechanism for lc.error_mech
    # - Use of AUTO_ASSIGN_MATRIX for the MappingProjections is safe, as compatibility of senders and receivers
    #    is checked in the instantiation of the learning_function

    error_source = lc.error_signal_mech

    learning_mechanism = LearningMechanism(default_variable=[activation_input,
                                                     activation_output,
                                                     error_signal],
                                           error_sources=error_source,
                                           function=learning_function,
                                           # learning_signals=[lc.activation_mech_projection],
                                           learning_signals=[learning_projection],
                                           name=LEARNING_MECHANISM + " for " + lc.activation_mech_projection.name)
    learning_mechanism.learning_type = LearningType.SUPERVISED
    learning_mechanism.learning_timing = LearningTiming.LEARNING_PHASE

    # IMPLEMENTATION NOTE:
    # ADD ARGUMENTS TO LearningMechanism FOR activation_input AND activation_output, AND THEN INSTANTIATE THE
    # MappingProjections BELOW IN CALL TO HELPER METHODS FROM LearningMechanism._instantiate_attributes_before_function
    # (FOLLOWING DESIGN OF _instantiate_error_signal_projection IN LearningMechanism):

    # Assign MappingProjection from activation_mech_input to LearningMechanism's ACTIVATION_INPUT inputPort
    lc._activation_mech_input_projection = MappingProjection(sender=lc.activation_mech_input,
                      receiver=learning_mechanism.input_ports[ACTIVATION_INPUT],
                      matrix=IDENTITY_MATRIX,
                      name = lc.activation_mech_input.owner.name + ' to ' + ACTIVATION_INPUT)

    # Assign MappingProjection from activation_mech_output to LearningMechanism's ACTIVATION_OUTPUT inputPort
    lc._activation_mech_output_projection = MappingProjection(sender=lc.activation_mech_output,
                      receiver=learning_mechanism.input_ports[ACTIVATION_OUTPUT],
                      matrix=IDENTITY_MATRIX,
                      name = lc.activation_mech_output.owner.name + ' to ' + ACTIVATION_OUTPUT)

    lc.learning_mechanism = learning_mechanism

    learning_projection._learning_components = lc


def _instantiate_error_signal_projection(sender, receiver):
    """Instantiate a MappingProjection to carry an error_signal to a LearningMechanism

    Can take as the sender an `ObjectiveMechanism` or a `LearningMechanism`.
    If the sender is an ObjectiveMechanism, uses its `primary OutputPort <OutputPort_Primary>`.
    If the sender is a LearningMechanism, uses its `ERROR_SIGNAL <LearningMechanism.output_ports>` OutputPort.
    The receiver must be a LearningMechanism; its `ERROR_SIGNAL <LearningMechanism.input_ports>` InputPort is used.
    Uses and IDENTITY_MATRIX for the MappingProjection, so requires that the sender be the same length as the receiver.

    """
    from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection

    if isinstance(sender, ObjectiveMechanism):
        sender = sender.output_ports[OUTCOME]
    elif isinstance(sender, LearningMechanism):
        sender = sender.output_ports[ERROR_SIGNAL]
    else:
        raise LearningAuxiliaryError("Sender of the error signal Projection {} "
                                     "must be either an ObjectiveMechanism or "
                                     "a LearningMechanism".format(sender))

    if isinstance(receiver, LearningMechanism):
        receiver = receiver.input_ports[ERROR_SIGNAL]
    else:
        raise LearningAuxiliaryError("Receiver of the error signal Projection "
                                     "{} must be a LearningMechanism".format(receiver))

    if len(sender.defaults.value) != len(receiver.defaults.value):
        raise LearningAuxiliaryError("The length of the OutputPort ({}) for "
                                     "the sender ({}) of the error signal "
                                     "Projection does not match the length of "
                                     "the InputPort ({}) for the receiver "
                                     "({})".format(len(sender.defaults.value),
                                                   sender.owner.name,
                                                   len(receiver.defaults.value),
                                                   receiver.owner.name))

    return MappingProjection(sender=sender,
                             receiver=receiver,
                             matrix=IDENTITY_MATRIX,
                             name=sender.owner.name + ' to ' + ERROR_SIGNAL)

@tc.typecheck
def _get_learning_mechanisms(mech:Mechanism, composition=None):
    """Return LearningMechanisms for all Projections to and from the specified Mechanism.

    If composition is specified, only LearningMechanisms for Projections to Mechanisms belonging to that composition
    (i.e., Process or System) are included in the list of LearningMechanisms returned

    Returns two lists:
        - one (aff) with the LearningMechanisms for the Mechanism's afferent projections
        - the other (eff) with the LearningMechanisms for Mechanism's efferent Projections
    """

    from psyneulink.core.components.system import System
    from psyneulink.core.components.process import Process
    aff = []
    eff = []

    if composition and not isinstance(composition, (Process, System)):
        raise LearningAuxiliaryError("composition argument for _get_learing_mechanisms ({}) must be a {} or {}".
                                     format(composition, Process.__name__, System.__name__))

    if isinstance(composition, Process):
        composition_attrib = 'processes'
    elif isinstance(composition, System):
        composition_attrib = 'systems'

    for projection in mech.path_afferents:
        if projection.has_learning_projection and (composition is None
                                                   or composition in projection.receiver.owner.processes
                                                   or composition in projection.receiver.owner.systems):

            aff.extend([learning_projection.sender.owner
                        for learning_projection in projection.parameter_ports[MATRIX].mod_afferents
                        if isinstance(learning_projection, LearningProjection)
                        and (not composition or composition in getattr(learning_projection.sender.owner,
                                                                       composition_attrib))
                        ])

    for projection in mech.efferents:
        if projection.has_learning_projection and (composition is None
                                                   or composition in projection.receiver.owner.processes
                                                   or composition in projection.receiver.owner.systems):
            eff.extend([learning_projection.sender.owner
                        for learning_projection in projection.parameter_ports[MATRIX].mod_afferents
                        if isinstance(learning_projection, LearningProjection)
                        and (not composition or composition in getattr(learning_projection.sender.owner,
                                                                       composition_attrib))
                       ])
    return aff, eff


def _assign_error_signal_projections(processing_mech:Mechanism,
                                     system,
                                     scope=None,
                                     objective_mech:tc.optional(ObjectiveMechanism)=None):
    """Assign appropriate error_signal Projections to LearningMechanisms for processing_mechanism's afferents.

    Assign an error_signal Projection to the LearningMechanism for each afferent Projection of processing_mechanism
    that is being learned, from the LearningMechanism of each of processing_mechanism's efferents that is being learned,
    unless such a projection already exists

    Scope can be a Process, System, or both (Process must be one in the System)_
    If scope is specified:
       - and it i
     [??and only for afferents and efferents that belong to the same System.]

    system argument is used to assign System to affected LearningMechanisms
    """

    # composition = None
    # if isinstance(scope, list):
    #     if len(scope)!=2:
    #         raise LearningAuxiliaryError("PROGRAM ERROR: Can only have two args: "
    #                                      "one must be System and the other a Proccess ({})".format(scope))
    #     for item in scope:
    #         if isinstance(item, Process):
    #             proc = item
    #         if isinstance(item, System):
    #             sys = item
    #         if not proc in sys.processes:
    #             raise LearningAuxiliaryError("PROGRAM ERROR: Proccess ({}) must be in System ({}) specified in scope" .
    #                                          format(proc, sys))
    # else:
    #     composition = scope
    composition = scope

    # Get all LearningMechanisms for Projection to and from sample_mech (processing_mechanism)
    afferent_learning_mechs, efferent_learning_mechs = _get_learning_mechanisms(processing_mech, scope)

    # For the LearningMechanism of each Projection to sample_mech that is being learned
    for aff_lm in afferent_learning_mechs:
        # Check that aff_lm receives in its ACTIVATION_OUTPUT InputPort the same Projection
        #    that the ObjectMechanism received
        if objective_mech and not (aff_lm.input_ports['activation_output'].path_afferents[0].sender ==
                                   objective_mech.input_ports[SAMPLE].path_afferents[0].sender):
            raise LearningAuxiliaryError("PROGRAM ERROR: The {} being assigned to replace {} ({}) receives its "
                                          "ACTIVATION_OUTPUT Projection from a source ({}) that is different than "
                                          "the source of the {}'s SAMPLE InputPort ({})."
                                         .format(LearningMechanism, objective_mech.name, aff_lm.name,
                                                  aff_lm.input_ports['activation_output'].path_afferents[0].sender,
                                                  aff_lm.name,
                                                  objective_mech.input_ports[SAMPLE].path_afferents[0].sender.name))
        # For each Projection from sample_mech that is being learned,
        #    add a Projection from its LearningMechanism ERROR_SIGNAL OutputPort
        #    to a newly created ERROR_SIGNAL InputPort on afferent_lm
        for eff_lm in efferent_learning_mechs:
            # Make sure Projection doesn't already exist
            if not any(proj.sender.owner == eff_lm for proj in aff_lm.afferents if ERROR_SIGNAL in proj.receiver.name):
                # aff_lm.add_ports(InputPort(variable=eff_lm.output_ports[ERROR_SIGNAL].value,
                #                              projections=eff_lm.output_ports[ERROR_SIGNAL],
                #                              name=ERROR_SIGNAL))
                aff_lm.add_ports(InputPort(projections=eff_lm.output_ports[ERROR_SIGNAL],
                                            name=ERROR_SIGNAL,
                                            context=Context(source=ContextFlags.METHOD)),
                                  context=Context(source=ContextFlags.METHOD))

        for projection in aff_lm.projections:
            projection._activate_for_compositions(system)

        if not aff_lm.systems:
            aff_lm._add_system(system, LEARNING)


class LearningComponents(object):
    """Gets components required to instantiate LearningMechanism and its Objective Function for a LearningProjection

    Has attributes for the following learning components relevant to a `LearningProjection`,
    each of which is found and/or validated if necessary before assignment:

    * `activation_mech_projection` (`MappingProjection`):  one being learned)
    * `activation_mech_input` (`OutputPort`):  input to Mechanism to which Projection being learned Projections
    * `activation_output_mech` (`ProcessingMechanism <ProcessingMechanism>`):  Mechanism to which projection being learned
                                                                        projects
    * `activation_mech_output` (`OutputPort`):  output of activation_output_mech
    * `activation_mech_fct` (function):  function of Mechanism to which projection being learned projects
    * `activation_derivative` (function):  derivative of function for activation_output_mech
    * `error_projection` (`MappingProjection`):  next projection in learning sequence after activation_mech_projection
    * `error_matrix` (`ParameterPort`):  parameterPort of error_projection with error_matrix
    * `error_derivative` (function):  deriviative of function of error_mech
    * `error_mech` (ProcessingMechanism):  Mechanism to which error_projection projects
    * `error_mech_output` (OutputPort):  outputPort of error_mech, that projects either to the next
                                          ProcessingMechanism in the pathway, or to an ObjectiveMechanism
    * `error_signal_mech` (LearningMechanism or ObjectiveMechanism):  Mechanism from which LearningMechanism
                                                                      gets its error_signal (ObjectiveMechanism for
                                                                      the last Mechanism in a learning sequence; next
                                                                      LearningMechanism in the sequence for all others)
    * `error_signal_mech_output` (OutputPort): outputPort of error_signal_mech, that projects to the preceeding
                                                LearningMechanism in the learning sequence (or nothing for the 1st mech)

    IMPLEMENTATION NOTE:  The helper methods in this class (that assign values to the various attributes)
                          respect membership in a process;  that is, they do not assign attribute values to
                          objects that belong to a process in which the root attribute (self.activation_output_mech)
                          belongs
    """

    def __init__(self, learning_projection, context=None):

        self._validate_learning_projection(learning_projection)
        self.learning_projection = learning_projection

        self._activation_mech_projection = None
        self._activation_output_mech = None
        self._activation_mech_input = None
        self._activation_mech_input_projection = None
        self._activation_mech_fct = None
        self._activation_derivative = None
        self._activation_mech_output = None
        self._activation_mech_output_projection = None
        self._error_projection = None
        self._error_matrix = None
        self._error_derivative = None
        self._error_mech = None
        self._error_mech_output = None
        self._error_signal_mech = None
        self._error_signal_mech_output = None
        # self._error_objective_mech = None
        # self._error_objective_mech_output = None


    def _validate_learning_projection(self, learning_projection):

        if not isinstance(learning_projection, LearningProjection):
            raise LearningAuxiliaryError("{} is not a LearningProjection".format(learning_projection.name))

    # HELPER METHODS FOR lc_components
    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_projection:  one being learned) (MappingProjection)
    @property
    def activation_mech_projection(self):
        def _get_act_proj():
            try:
                self.activation_mech_projection = self.learning_projection.receiver.owner
                return self.learning_projection.receiver.owner
            except AttributeError:
                raise LearningAuxiliaryError("activation_mech_projection not identified: learning_projection ({})"
                                              "not appear to have been assiged a receiver.".
                                             format(self.learning_projection))
        return self._activation_mech_projection or _get_act_proj()

    @activation_mech_projection.setter
    def activation_mech_projection(self, assignment):
        if isinstance(assignment, (MappingProjection)):
            self._activation_mech_projection = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to activation_mech_projection; "
                                          "it must be a MappingProjection.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_output_mech:  mechanism to which projection being learned projects (ProcessingMechanism)
    @property
    def activation_output_mech(self):
        def _get_act_mech():
            if not self.activation_mech_projection:
                return None
            try:
                self.activation_output_mech = self.activation_mech_projection.receiver.owner
                return self.activation_mech_projection.receiver.owner
            except AttributeError:
                raise LearningAuxiliaryError("activation_output_mech not identified: activation_mech_projection ({})"
                                              "not appear to have been assiged a receiver.".
                                             format(self.learning_projection))
        return self._activation_output_mech or _get_act_mech()

    @activation_output_mech.setter
    def activation_output_mech(self, assignment):
        if isinstance(assignment, (ProcessingMechanism_Base)):
            self._activation_output_mech = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to activation_output_mech; "
                                          "it must be a ProcessingMechanism.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_input:  input to mechanism to which projection being learned projections (OutputPort)
    @property
    def activation_mech_input(self):
        def _get_act_input():
            if not self.activation_mech_projection:
                return None
            try:
                self.activation_mech_input = self.activation_mech_projection.sender
                return self.activation_mech_projection.sender
            except AttributeError:
                raise LearningAuxiliaryError("activation_mech_input not identified: activation_mech_projection ({})"
                                              "not appear to have been assiged a sender.".
                                             format(self.activation_mech_projection))
        return self._activation_mech_input or _get_act_input()

    @activation_mech_input.setter
    def activation_mech_input(self, assignment):
        if isinstance(assignment, (OutputPort)):
            self._activation_mech_input = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to activation_mech_input; "
                                          "it must be a OutputPort.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_fct:  function of mechanism to which projection being learned projects (function)
    @property
    def activation_mech_fct(self):
        def _get_act_mech_fct():
            if not self.activation_output_mech:
                return None
            try:
                self.activation_mech_fct = self.activation_output_mech.function
                return self.activation_output_mech.function
            except AttributeError:
                raise LearningAuxiliaryError("activation_mech_fct not identified: activation_output_mech ({})"
                                              "not appear to have been assiged a Function.".
                                             format(self.learning_projection))
        return self._activation_mech_fct or _get_act_mech_fct()

    @activation_mech_fct.setter
    def activation_mech_fct(self, assignment):
        if isinstance(assignment, (Function)):
            self._activation_mech_fct = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to activation_mech_fct; "
                                          "it must be a Function.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_derivative:  derivative of function for activation_output_mech (function)
    @property
    def activation_derivative(self):
        def _get_act_deriv():
            if not self.activation_mech_fct:
                return None
            try:
                self._activation_derivative = self.activation_output_mech.function.derivative
                return self.activation_output_mech.function.derivative
            except AttributeError:
                raise LearningAuxiliaryError("activation_derivative not identified: activation_mech_fct ({})"
                                              "not appear to have a derivative defined.".
                                             format(self.learning_projection))
        return self._activation_derivative or _get_act_deriv()

    @activation_derivative.setter
    def activation_derivative(self, assignment):
        if isinstance(assignment, (types.FunctionType, types.MethodType)):
            self._activation_derivative = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to activation_derivative; "
                                          "it must be a function or method.")


    # ---------------------------------------------------------------------------------------------------------------
    # activation_mech_output:  output of activation_output_mech (OutputPort)
    @property
    def activation_mech_output(self):
        def _get_act_sample():
            if not self.activation_output_mech:
                return None
            # If MONITOR_FOR_LEARNING specifies an outputPort, use that
            try:
                sample_port_Name = self.activation_output_mech.monitor_for_learning
                self.activation_mech_output = self.activation_output_mech.output_ports[sample_port_Name]
                if not isinstance(self.activation_mech_output, OutputPort):
                    raise LearningAuxiliaryError("The specification of the MONITOR_FOR_LEARNING parameter ({}) "
                                                  "for {} is not an outputPort".
                                                 format(self.activation_mech_output, self.learning_projection))
            except KeyError:
                # No outputPort specified so use primary outputPort
                try:
                    self.activation_mech_output = self.activation_output_mech.output_port
                except AttributeError:
                    raise LearningAuxiliaryError("activation_mech_output not identified: activation_output_mech ({})"
                                                  "not appear to have been assigned a primary outputPort.".
                                                 format(self.learning_projection))
            return self.activation_output_mech.output_port

        return self._activation_mech_output or _get_act_sample()

    @activation_mech_output.setter
    def activation_mech_output(self, assignment):
        if isinstance(assignment, (OutputPort)):
            self._activation_mech_output =assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to activation_mech_output; "
                                          "it must be a OutputPort.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_matrix:  parameterPort for error_matrix (ParameterPort)
    # This must be found
    @property
    def error_matrix(self):

        # Find from error_projection
        def _get_err_matrix():
            if not self.error_projection:
                return None
            try:
                self.error_matrix = self.error_projection._parameter_ports[MATRIX]
                return self.error_projection._parameter_ports[MATRIX]
            except AttributeError:
                raise LearningAuxiliaryError("error_matrix not identified: error_projection ({})"
                                              "not not have a {} parameterPort".
                                             format(self.error_projection))
        return self._error_matrix or _get_err_matrix()

    @error_matrix.setter
    def error_matrix(self, assignment):
        if isinstance(assignment, (ParameterPort)):
            self._error_matrix = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to error_matrix; "
                                          "it must be a ParameterPort.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_projection:  one that has the error_matrix (MappingProjection)
    @property
    def error_projection(self):
        # Find from activation_output_mech:
        def _get_error_proj():
            if not self.activation_mech_output:
                return None
            projections = self.activation_mech_output.efferents
            # error_proj must be a MappingProjection that:
            #    - projects to another Mechanism in the same Process as the one that projects to it and
            #    - has a LearningProjection to it
            error_proj = next((projection for projection in projections if
                               (isinstance(projection, MappingProjection) and
                                projection.has_learning_projection and
                                any(process in projection.receiver.owner.processes
                                    # for process in self.activation_output_mech.processes))),
                                    for process in self.activation_mech_input.owner.processes))),
                              None)
            if not error_proj:
                return None
            self.error_projection = error_proj
            return error_proj
        return self._error_projection or _get_error_proj()

    @error_projection.setter
    def error_projection(self, assignment):
        if isinstance(assignment, (MappingProjection)):
            self._error_projection = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to error_projection; "
                                          "it must be a MappingProjection.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_mech:  mechanism to which error_projection projects (ProcessingMechanism)
    @property
    def error_mech(self):

        # Find from error_projection
        def _get_err_mech():
            if not self.error_projection:
                return None
            try:
                self.error_mech = self.error_projection.receiver.owner
            except AttributeError:
                raise LearningAuxiliaryError("error_mech not identified: error_projection ({})"
                                              "does not appear to have a receiver or owner".
                                             format(self.error_projection))
            if not isinstance(self.error_mech, ProcessingMechanism_Base):
                raise LearningAuxiliaryError("error_mech found ({}) but it does not "
                                              "appear to be a ProcessingMechanism".
                                             format(self.error_mech.name))
            return self.error_projection.receiver.owner
        return self._error_mech or _get_err_mech()

    @error_mech.setter
    def error_mech(self, assignment):
        if isinstance(assignment, (ProcessingMechanism_Base)):
            self._error_mech = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to error_mech; "
                                          "it must be a ProcessingMechanism.")


    # -----------------------------------------------------------------------------------------------------------
    # error_derivative:  derivative of function of error_mech (function)
    @property
    def error_derivative(self):
        # Find from error_mech:
        def _get_error_deriv():
            if not self.error_mech:
                return None
            try:
                self.error_derivative = self.error_mech.function.derivative
                return self.error_mech.function.derivative
            except AttributeError:
                raise LearningAuxiliaryError("error_derivative not identified: the function ({}) "
                                              "for error_mech ({}) does not have a derivative attribute".
                                             format(self.error_mech.function.__class__.__name__,
                                                     self.error_mech.name))
        return self._error_derivative or _get_error_deriv()

    @error_derivative.setter
    def error_derivative(self, assignment):
        if isinstance(assignment, (types.FunctionType, types.MethodType)):
            self._error_derivative = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to error_derivative; "
                                          "it must be a function.")


    # ---------------------------------------------------------------------------------------------------------------
    # error_mech_output: outputPort of mechanism to which error_projection projects (OutputPort)
    @property
    def error_mech_output(self):
        # Find from error_mech
        def _get_err_mech_out():
            if not self.error_mech:
                return None
            try:
                self.error_mech_output = self.error_mech.output_port
            except AttributeError:
                raise LearningAuxiliaryError("error_mech_output not identified: error_mech ({})"
                                              "does not appear to have an outputPort".
                                             format(self.error_mech_output))
            if not isinstance(self.error_mech_output, OutputPort):
                raise LearningAuxiliaryError("error_mech_output found ({}) but it does not "
                                              "appear to be an OutputPort".
                                             format(self.error_mech_output.name))
            return self.error_mech.output_port
        return self._error_mech_output or _get_err_mech_out()

    @error_mech_output.setter
    def error_mech_output(self, assignment):
        if isinstance(assignment, (OutputPort)):
            self._error_mech_output = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to error_mech_output; "
                                          "it must be an OutputPort.")

    # ---------------------------------------------------------------------------------------------------------------
    # error_signal_mech:  learning mechanism for error_projection (LearningMechanism)
    @property
    def error_signal_mech(self):
        # Find from error_matrix:
        def _get_error_signal_mech():
            if not self.error_matrix:
                return None
            # search the projections to the error_matrix ParameterPort for a LearningProjection
            learning_proj = next((proj for proj in self.error_matrix.mod_afferents
                                 if isinstance(proj, LearningProjection)), None)
            # if there are none, the error_matrix might be for an error_projection to an ObjectiveMechanism
            #   (i.e., the TARGET mechanism)
            if not learning_proj:
                # if error_mech is the last in the learning sequence, then its error_matrix does not receive a
                #    LearningProjection, but its error_projection does project to an ObjectiveMechanism, so return that
                objective_mechanism = self.error_matrix.owner.receiver.owner
                if not isinstance(objective_mechanism, ObjectiveMechanism):
                    raise LearningAuxiliaryError("error_signal_mech not identified: error_matrix does not have "
                                                  "a LearningProjection and error_projection does not project to a "
                                                  "TARGET ObjectiveMechanism")
                else:
                    self.error_signal_mech = objective_mechanism
                    return self.error_signal_mech
            try:
                learning_mech = learning_proj.sender.owner
            except AttributeError:
                raise LearningAuxiliaryError("error_signal_mech not identified: "
                                              "the LearningProjection to error_matrix does not have a sender")
            if not isinstance(learning_mech, LearningMechanism):
                raise LearningAuxiliaryError("error_signal_mech not identified: "
                                              "the LearningProjection to error_matrix does not come from a "
                                              "LearningMechanism")

            # Get the current Process (the one to which the current and preceding Mechanism in the sequence belong)
            current_processes = (set(self.activation_output_mech.processes.keys()).
                                 intersection(set(self.activation_mech_input.owner.processes.keys())))

            # If the learning_mech found has already been assigned to a Process that is not the current Process,
            #    then the current mechanism (self.activation_output_mech) may:
            #    - be the TERMINAL for its process,
            #    - or it may project to the same next Mechanism as the other Process

            # if learning_mech.processes and not any(process in self.activation_mech_input.owner.processes
            #                                        for process in learning_mech.processes):
            if learning_mech.processes and not any(process in current_processes
                                                   for process in learning_mech.processes):
               # Look for an ObjectiveMechanism that projects to the current Mechanism (self.activation_output_mech)
                objective_learning_proj = next((projection for projection in
                                                self.activation_output_mech.output_port.efferents
                                                if isinstance(projection.receiver.owner, ObjectiveMechanism)), None)
                # Return ObjectiveMechanism found as error_signal_mech
                if objective_learning_proj:
                    learning_mech = objective_learning_proj.receiver.owner

                # No ObjectiveMechanism was found,
                #    so check if error_projection projects to another Mechanism in the same Process:
                #        - if it doesn't, throw an exception
                #        - if it does, assign learning_mech to any shared Processes, and return as error_signal_mech
                # elif any(process in learning_proj.receiver.owner.receiver.owner.processes
                else:
                    next_mech_processes = set(self.error_projection.receiver.owner.processes.keys())
                    shared_processes = current_processes.intersection(next_mech_processes)
                    if not shared_processes:
                        raise LearningAuxiliaryError("PROGRAM ERROR: expected to find that {} projects "
                                                      "to another Mechanism in {}".
                                                     format(self.activation_output_mech.name, current_processes))
                    for process in shared_processes:
                        if not process in learning_mech.processes:
                            learning_mech._add_process(process, LEARNING)

            self.error_signal_mech = learning_mech
            return self.error_signal_mech

        return self._error_signal_mech or _get_error_signal_mech()

    @error_signal_mech.setter
    def error_signal_mech(self, assignment):
        if (assignment is None or
                isinstance(assignment, LearningMechanism) or
                (isinstance(assignment, ObjectiveMechanism) and assignment._role is LEARNING)):
            self._error_signal_mech = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment "
                                          "to error_signal_mech; it must be a "
                                          "LearningMechanism.")

    # ---------------------------------------------------------------------------------------------------------------
    # FIX: MODIFY TO RETURN EITHER outputPort if it is an ObjectiveMechanism or
    #                              output_ports[ERROR_SIGNAL] if it it a LearningMechanism
    # error_signal_mech_output: outputPort of LearningMechanism for error_projection (OutputPort)
    @property
    def error_signal_mech_output(self):
        # Find from error_mech
        def _get_err_sig_mech_out():
            if not self.error_signal_mech:
                return None
            if isinstance(self.error_signal_mech, ObjectiveMechanism):
                try:
                    self.error_signal_mech_output = self.error_signal_mech.output_port
                except AttributeError:
                    raise LearningAuxiliaryError("error_signal_mech_output "
                                                  "not identified: "
                                                  "error_signal_mech ({}) does "
                                                  "not appear to have an "
                                                  "OutputPort".
                                                 format(self.error_signal_mech.name))
                if not isinstance(self.error_signal_mech_output, OutputPort):
                    raise LearningAuxiliaryError("error_signal_mech_output "
                                                  "found ({}) for {} but it "
                                                  "does not appear to be an "
                                                  "OutputPort".
                                                 format(self.error_signal_mech.name,
                                                         self.error_signal_mech_output.name))
            elif isinstance(self.error_signal_mech, LearningMechanism):
                try:
                    self.error_signal_mech_output = self.error_signal_mech.output_ports[ERROR_SIGNAL]
                except AttributeError:
                    raise LearningAuxiliaryError("error_signal_mech_output "
                                                  "not identified: "
                                                  "error_signal_mech ({}) does "
                                                  "not appear to have an "
                                                  "ERROR_SIGNAL outputPort".
                                                 format(self.error_signal_mech))
                if not isinstance(self.error_signal_mech_output, OutputPort):
                    raise LearningAuxiliaryError("error_signal_mech_output "
                                                  "found ({}) for {} but it "
                                                  "does not appear to be an "
                                                  "OutputPort".format(self.error_signal_mech.name,
                                                                       self.error_signal_mech_output.name))
            return self.error_signal_mech.output_port
        return self._error_signal_mech_output or _get_err_sig_mech_out()

    @error_signal_mech_output.setter
    def error_signal_mech_output(self, assignment):
        if isinstance(assignment, (OutputPort)):
            self._error_signal_mech_output = assignment
        else:
            raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment "
                                          "to error_signal_mech_output; it "
                                          "must be an OutputPort.")

    # # ---------------------------------------------------------------------------------------------------------------
    # # error_objective_mech:  TARGET objective mechanism for error_mech (ObjectiveMechanism)
    # @property
    # def error_objective_mech(self):
    #     # Find from error_matrix:
    #     def _get_obj_mech():
    #         if not self.error_matrix:
    #             return None
    #         learning_proj = next((proj for proj in self.error_matrix.path_afferents
    #                              if isinstance(proj, LearningProjection)),None)
    #         if not learning_proj:
    #             # error_matrix is for a MappingProjection that projects to the TARGET ObjectiveMechanism, so return that
    #             if isinstance(self.error_matrix.owner.receiver.owner, ObjectiveMechanism):
    #                 self.error_objective_mech = self.error_matrix.owner.receiver.owner
    #                 return self.error_matrix.owner.receiver.owner
    #             else:
    #                 raise LearningAuxiliaryError("error_objective_mech not identified: error_matrix does not have a "
    #                                               "LearningProjection and is not for a TARGET ObjectiveMechanism")
    #         try:
    #             learning_mech = learning_proj.sender.owner
    #         except AttributeError:
    #             raise LearningAuxiliaryError("error_objective_mech not identified: "
    #                                           "the LearningProjection to error_matrix does not have a sender")
    #         if not isinstance(learning_mech, LearningMechanism):
    #             raise LearningAuxiliaryError("error_objective_mech not identified: "
    #                                           "the LearningProjection to error_matrix does not come from a "
    #                                           "LearningMechanism")
    #         try:
    #             error_obj_mech = next((proj.sender.owner
    #                                    for proj in learning_mech.input_ports[ERROR_SIGNAL].path_afferents
    #                                    if isinstance(proj.sender.owner, ObjectiveMechanism)),None)
    #         except AttributeError:
    #             # return None
    #             raise LearningAuxiliaryError("error_objective_mech not identified: "
    #                                           "could not find any projections to the LearningMechanism ({})".
    #                                           format(learning_mech))
    #         # if not error_obj_mech:
    #         #     raise LearningAuxiliaryError("error_objective_mech not identified: "
    #         #                                   "the LearningMechanism ({}) does not receive a projection "
    #         #                                   "from an ObjectiveMechanism".
    #         #                                   format(learning_mech))
    #         self.error_objective_mech = error_obj_mech
    #         return error_obj_mech
    #
    #     return self._error_objective_mech or _get_obj_mech()
    #
    # @error_objective_mech.setter
    # def error_objective_mech(self, assignment):
    #     if assignment is None or isinstance(assignment, (ObjectiveMechanism)):
    #         self._error_objective_mech = assignment
    #     else:
    #         raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to error_objective_mech; "
    #                                       "it must be an ObjectiveMechanism.")
    #
    # # ---------------------------------------------------------------------------------------------------------------
    # # error_objective_mech_output: outputPort of ObjectiveMechanism for error_projection (ObjectiveMechanism)
    # @property
    # def error_objective_mech_output(self):
    #     # Find from error_mech
    #     def _get_err_obj_mech_out():
    #         if not self.error_objective_mech:
    #             return None
    #         try:
    #             self.error_objective_mech_output = self.error_objective_mech.outputPort
    #         except AttributeError:
    #             raise LearningAuxiliaryError("error_objective_mech_output not identified: error_objective_mech ({})"
    #                                           "does not appear to have an outputPort".
    #                                           format(self.error_objective_mech_output))
    #         if not isinstance(self.error_objective_mech_output, OutputPort):
    #             raise LearningAuxiliaryError("error_objective_mech_output found ({}) but it does not "
    #                                           "appear to be an OutputPort".
    #                                           format(self.error_objective_mech_output.name))
    #         return self.error_objective_mech.outputPort
    #     return self._error_objective_mech_output or _get_err_obj_mech_out()
    #
    # @error_objective_mech_output.setter
    # def error_objective_mech_output(self, assignment):
    #     if isinstance(assignment, (OutputPort)):
    #         self._error_objective_mech_output = assignment
    #     else:
    #         raise LearningAuxiliaryError("PROGRAM ERROR: illegal assignment to error_objective_mech_output; "
    #                                       "it must be an OutputPort.")
