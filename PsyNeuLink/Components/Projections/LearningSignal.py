# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningSignal **********************************************************

"""
.. _LearningSignal_Overview:

Overview
--------

A LearningSignal projection takes a value (an *error signal*) from a :doc:`MonitoringMechanism` (its ``sender``),
and uses this to compute a ``weightChangeMatrix`` that is assigned as its value.  This is used to modify the
:ref:`matrix <Mapping_Matrix>` parameter of a :doc:`Mapping` projection.  A LearningSignal can be assigned different
functions[LINK] to implement different learning algorithms, which are associated with corresponding types of
MonitoringMechanisms.

.. _LearningSignal_Creating_A_LearningSignal_Projection:

Creating a LearningSignal Projection
------------------------------------

???INCLUDE: automatically by specifying the LEARNING_SIGNAL parameter of a Mapping Projection


A LearningSignal projection can be created in any of the ways that can be used to
:ref:`create a projection <Projection_Creating_A_Projection>`, or by including it in the specification of a
:ref:`system <System>`, :ref:`process <Process>`, or projection in the :ref:`pathway <>` of a process.  Its
``sender`` (the source of its error signal) must be a MonitoringMechanism, and its ``receiver`` must be the
parameterState of a Mapping projection.  When a LearningSignal is created, its full initialization is deferred [LINK]
until its ``sender`` and ``receiver`` have been fully specified.  This means that it is possible to create a
LearningSignal using its constructor without specifying either its ``sender`` or its ``receiver``.

COMMENT:
   CURRENT
COMMENT
It is not necessary to assign a ``sender``;  if none is specified when it is initialized, a MonitoringMechanism of
the appropriate type will be created (see :ref:`Structure <LearningSignal_Structure>` below).  However,
a LearningSignal's ``receiver`` must be specified.  One that is done, for the LearningSignal to be operational,
initializaton must be completed by calling its ``deferred_init`` method.  This is not necessary if the LearningSignal
is specified as part of a system, process, or a projection in the pathway of a process -- in those cases,
initialization is completed automatically.

THIS IS A TEST

COMMENT:
   REPLACE WITH THIS ONCE FUNCTIONALITY IS IMPLEMENTED
    Initialization will
    be  completed as soon as the LearningSignal has been assigned as the projection *from* a MonitoringMechanism (i.e.,
    it's ``sender`` has been specified) and as the projection *to* the parameterState of a Mapping projection (i.e.,
    its ``receiver`` has been specified).  This is handled automatically when the LearningSignal is specified as part of
    a system, process, or projection in the pathway of a process.
COMMENT

.. _LearningSignal_Structure:

Structure
---------

POINT TO FIGURE IN PROCESS

*Error Signal*.  The ``variable`` for a LearningSignal is used as its ``errorSignal``.  It uses this to compute the
changes to the ``matrix`` parameter of the Mapping projection for which it is responsible (i.e., to which it
projects).  The error signal comes from the LearningSignal's ``sender``, which is a MonitoringMechanism that
evaluates the output of the ProcessingMechanism that receives the Mapping projection being trained (see :ref:`figure
<Process_Learning_Figure>`).


and is used
an error signal
, when it is executed, must have three components

# DOCUMENT: self.variable has 3 items:
#   input: <Mapping projection>.sender.value
#              (output of the Mechanism that is the sender of the Mapping projection)
#   output: <Mapping projection>.receiver.owner.outputState.value == self.errorSource.outputState.value
#              (output of the Mechanism that is the receiver of the Mapping projection)
#   error_signal: <Mapping projection>.receiver.owner.parameterState[MATRIX].receivesFromProjections[0].sender.value ==
#                 self.errorSource.monitoringMechanism.value
#                 (output of the MonitoringMechanism that is the sender of the LearningSignal
#                  for the next Mapping projection in the Process)

*Function*.

# The default for FUNCTION is BackPropagation;  also Reinforcement -- each requires a specific types of
ProcessingMechanisms and configuration of its MOnitoringMechanisms

*Monitoring Mechanism*.  -> error source

Primary (Comparator) vs. seconadry (WeightedError)

# DOCUMENT: if it instantiates a DefaultTrainingSignal:
#               if outputState for error source is specified in its paramsCurrent[MONITOR_FOR_LEARNING], use that
#               otherwise, use error_source.outputState (i.e., error source's primary outputState)



.. _LearningSignal_Execution:

Execution
---------

A LearningSignal projection uses its ``function`` to compute its ``intensity``, and its :ref:`cost functions
<LearningSignal_Cost_Functions> use the ``intensity`` to compute the its ``cost``.  The ``intensity`` is assigned as
the LearningSignal projection's ``value``, which is used by the parmaterState to which it projects to modify the
corresponding parameter of the owner mechanism's function.

.. note::
   The changes in a parameter in response to the execution of a LearningSignal projection are not applied until the
   mechanism that receives the projection are next executed; see Lazy_Evaluation for an explanation of "lazy"
   updating).

.. _LearningSignal_Class_Reference:


Class Reference
---------------

"""


from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.Comparator import Comparator, COMPARATOR_SAMPLE
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import MonitoringMechanism_Base
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.WeightedError import WeightedError, NEXT_LEVEL_PROJECTION
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.Projections.Mapping import Mapping
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.Functions.Function import BackPropagation

# Params:

kwWeightChangeParams = "weight_change_params"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

DefaultTrainingMechanism = Comparator

class LearningSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningSignal(Projection_Base):
    """Implements projection that modifies the matrix param of a Mapping projection

    COMMENT:
        Description:
            The LearningSignal class is a componentType in the Projection category of Function,
            It's execute method takes the output of a MonitoringMechanism (self.variable), and the input and output of
                the ProcessingMechanism to which its receiver Mapping Projection projects, and generates a matrix of
                weight changes for the Mapping Projection's matrix parameter

        Class attributes:
            + className = LEARNING_SIGNAL
            + componentType = PROJECTION
            # + defaultSender (State)
            # + defaultReceiver (State)
            + paramClassDefaults (dict):
                + FUNCTION (Function): (default: BP)
                + FUNCTION_PARAMS:
                    + LEARNING_RATE (value): (default: 1)
            + paramNames (dict)
            + classPreference (PreferenceSet): LearningSignalPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            function (computes function specified in params[FUNCTION]
    COMMENT

    Arguments
    ---------
    - sender (MonitoringMechanism) - source of projection input (default: TBI)
    - receiver: (Mapping Projection) - destination of projection output (default: TBI)
    - params (dict) - dictionary of projection params:
        + FUNCTION (Function): (default: BP)
        + FUNCTION_PARAMS (dict):
            + LEARNING_RATE (value): (default: 1)
    - name (str) - if it is not specified, a default based on the class is assigned in register_category
    - prefs (PreferenceSet or specification dict):
         if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
         dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
         (see Description under PreferenceSet for details)


    Attributes
    ----------
    + sender (MonitoringMechanism)
    + receiver (Mapping)
    + paramInstanceDefaults (dict) - defaults for instance (created and validated in Components init)
    + paramsCurrent (dict) - set currently in effect
    + variable (value) - used as input to projection's function
    + value (value) - output of function
    + mappingWeightMatrix (2D np.array) - points to <Mapping>.paramsCurrent[FUNCTION_PARAMS][MATRIX]
    + weightChangeMatrix (2D np.array) - rows:  sender deltas;  columns:  receiver deltas
    + errorSignal (1D np.array) - sum of errors for each sender element of Mapping projection
    + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
    + prefs (PreferenceSet) - if not specified as an arg, default is created by copying LearningSignalPreferenceSet

    """

    componentType = LEARNING_SIGNAL
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_SENDER: MonitoringMechanism_Base,
                               PARAMETER_STATES: None, # This suppresses parameterStates
                               kwWeightChangeParams:  # Determine how weight changes are applied to weight matrix
                                   {                  # Note:  assumes Mapping.function is LinearCombination
                                       FUNCTION_PARAMS: {OPERATION: SUM},
                                       PARAMETER_MODULATION_OPERATION: ModulationOperation.ADD,
                                       PROJECTION_TYPE: LEARNING_SIGNAL}
                               })

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 function=BackPropagation(learning_rate=1,
                                          activation_function=Logistic()),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
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
        params = self._assign_args_to_param_dicts(function=function, params=params)

        # Store args for deferred initialization
        self.init_args = locals().copy()
        self.init_args['context'] = self
        self.init_args['name'] = name

        # Flag for deferred initialization
        self.value = DEFERRED_INITIALIZATION

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Insure sender is a MonitoringMechanism or ProcessingMechanism and receiver is a ParameterState or Mapping

        Validate send in params[PROJECTION_SENDER] or, if not specified, sender arg:
        - must be the outputState of a MonitoringMechanism (e.g., Comparator or WeightedError)
        - must be a list or 1D np.array (i.e., the format of an errorSignal)

        Validate receiver in params[PARAMETER_STATES] or, if not specified, receiver arg:
        - must be either a Mapping projection or parameterStates[MATRIX]

         """

        # Parse params[PROJECTION_SENDER] if specified, and assign self.sender
        super()._validate_params(request_set, target_set, context)

        # VALIDATE SENDER
        sender = self.sender
        self.validate_sender(sender)

        # VALIDATE RECEIVER
        try:
            receiver = target_set[PARAMETER_STATES]
            self.validate_receiver(receiver)
        except (KeyError, LearningSignalError):
            # PARAMETER_STATES not specified:
            receiver = self.receiver
            self.validate_receiver(receiver)

        # VALIDATE WEIGHT CHANGE PARAMS
        try:
            weight_change_params = target_set[kwWeightChangeParams]
        except KeyError:
            pass
        else:
            # FIX: CHECK THAT EACH ONE INCLUDED IS A PARAM OF A LINEAR COMBINATION FUNCTION
            for param_name, param_value in weight_change_params.items():
                if param_name is FUNCTION:
                    raise LearningSignalError("{} of {} contains a function specification ({}) that would override"
                                              " the LinearCombination function of the targetted mapping projection".
                                              format(kwWeightChangeParams,
                                                     self.name,
                                                     param_value))

    def validate_sender(self, sender):
        """Make sure sender is a MonitoringMechanism or ProcessingMechanism or the outputState for one;
        """

        # If specified as a MonitoringMechanism, reassign to its outputState
        if isinstance(sender, MonitoringMechanism_Base):
            sender = self.sender = sender.outputState

        # If it is the outputState of a MonitoringMechanism, check that it is a list or 1D np.array
        if isinstance(sender, OutputState):
            if not isinstance(sender.value, (list, np.ndarray)):
                raise LearningSignalError("Sender for {} (outputState of MonitoringMechanism {}) "
                                          "must be a list or 1D np.array".format(self.name, sender))
            if not np.array(sender.value).ndim == 1:
                raise LearningSignalError("OutputState of MonitoringMechanism ({}) for {} must be an 1D np.array".
                                          format(sender, self.name))

        # If specification is a MonitoringMechanism class, pass (it will be instantiated in _instantiate_sender)
        elif inspect.isclass(sender) and issubclass(sender,  MonitoringMechanism_Base):
            pass

        else:
            raise LearningSignalError("Sender arg (or {} param ({}) for must be a MonitoringMechanism, "
                                      "the outputState of one, or a reference to the class"
                                      .format(PROJECTION_SENDER, sender, self.name, ))

    def validate_receiver(self, receiver):
        """Make sure receiver is a Mapping projection or the parameterState of one
        """

        if not isinstance(receiver, (Mapping, ParameterState)):
            raise LearningSignalError("Receiver arg ({}) for {} must be a Mapping projection or a parameterState of one"
                                      .format(receiver, self.name))
        # If it is a parameterState and the receiver already has a parameterStates dict
        #     make sure the assignment is to its MATRIX entry
        if isinstance(receiver, ParameterState):
            if receiver.owner.parameterStates and not receiver is receiver.owner.parameterStates[MATRIX]:
                raise LearningSignalError("Receiver arg ({}) for {} must be the {} parameterState of a"
                                          "Mapping projection".format(receiver, self.name, MATRIX, ))

        # Notes:
        # * if specified as a Mapping projection, it will be assigned to a parameter state in _instantiate_receiver
        # * the value of receiver will be validated in _instantiate_receiver

    def _instantiate_attributes_before_function(self, context=None):
        """Override super to call _instantiate_receiver before calling _instantiate_sender

        Call _instantiate_receiver first since both _instantiate_sender and _instantiate_function
            reference the Mapping projection's weight matrix: self.mappingProjection.matrix

        """
        # FIX: PROBLEM: _instantiate_receiver usually follows _instantiate_function,
        # FIX:          and uses self.value (output of function) to validate against receiver.variable

        self._instantiate_receiver(context)

        # # MODIFIED 8/14/16: COMMENTED OUT SINCE SOLVED BY MOVING add_to TO _instantiate_attributes_after_function
        # # "Cast" self.value to Mapping Projection parameterState's variable to pass validation in _instantiate_sender
        # # Note: this is because _instantiate_sender calls _add_projection_to
        # # (since self.value is not assigned until _instantiate_function; it will be reassigned there)
        # self.value = self.receiver.variable

        super()._instantiate_attributes_before_function(context)

    def _instantiate_attributes_after_function(self, context=None):
        """Override super since it calls _instantiate_receiver which has already been called above
        """
        # pass
        # MODIFIED 8/14/16: MOVED FROM _instantiate_sender
        # Add LearningSignal projection to Mapping projection's parameterState
        # Note: needs to be done after _instantiate_function, since validation requires self.value be assigned
        self.add_to(receiver=self.mappingProjection, state=self.receiver, context=context)

    def _instantiate_receiver(self, context=None):
        """Instantiate and/or assign the parameterState of the projection to be modified by learning

        If receiver is specified as a Mapping Projection, assign LearningSignal to parameterStates[MATRIX]
            for the projection;  if that does not exist, instantiate and assign as the receiver for the LearningSignal
        If specified as a ParameterState, validate that it is parameterStates[MATRIX]
        Validate that the LearningSignal's error matrix is the same shape as the recevier's weight matrix
        Re-assign LearningSignal's variable to match the height (number of rows) of the matrix
        
        Notes:
        * This must be called before _instantiate_sender since that requires access to self.receiver
            to determine whether to use a comparator mechanism or <Mapping>.receiverError for error signals
        * Doesn't call super()._instantiate_receiver since that assumes self.receiver.owner is a Mechanism
                              and calls _add_projection_to_mechanism
        """

# FIX: ??REINSTATE CALL TO SUPER AFTER GENERALIZING IT TO USE Projection.add_to
# FIX: OR, MAKE SURE FUNCTIONALITY IS COMPARABLE

        weight_change_params = self.paramsCurrent[kwWeightChangeParams]

        # VALIDATE that self.receiver is a ParameterState or a Mapping Projection

        # If receiver is a ParameterState, and receiver's parameterStates dict has been instantiated,
        #    make sure LearningSignal is being assigned to the parameterStates[MATRIX] of a Mapping projection
        if isinstance(self.receiver, ParameterState):

            self.mappingProjection = self.receiver.owner

            # MODIFIED 10/29/16 OLD:
            # Reciever must be a Mapping projection with a LinearCombination function
            if not isinstance(self.mappingProjection, Mapping):
                raise LearningSignalError("Receiver arg ({}) for {} must be the parameterStates[{}] "
                                          "of a Mapping (rather than a {})".
                                          format(self.receiver,
                                                 self.name,
                                                 MATRIX,
                                                 self.mappingProjection.__class__.__name__))
            if not isinstance(self.receiver.function.__self__, LinearCombination):
                raise LearningSignalError("Function of receiver arg ({}) for {} must be a {} (rather than {})".
                                          format(self.receiver,
                                                 self.name,
                                                 kwLinearCombination,
                                                 self.mappingProjection.function.__self__.__class__.__name__))

            # # MODIFIED 10/29/16 NEW:
            # # Reciever must be the parameterState for a Mapping projection with a LinearCombination identity function
            # if not isinstance(self.mappingProjection, Mapping):
            #     raise LearningSignalError("Receiver arg ({}) for {} must be the parameterStates[{}] "
            #                               "of a Mapping (rather than a {})".
            #                               format(self.receiver,
            #                                      self.name,
            #                                      MATRIX,
            #                                      self.mappingProjection.__class__.__name__))
            # if not isinstance(self.receiver.function.__self__, LinearCombination):
            #     raise LearningSignalError("Function of receiver arg ({}) for {} must be a {} (rather than {})".
            #                               format(self.receiver,
            #                                      self.name,
            #                                      kwLinear,
            #                                      self.mappingProjection.function.__self__.__class__.__name__))
            # # MODIFIED 10/29/16 END


            # receiver is parameterState[MATRIX], so update its params with ones specified by LearningSignal
            # (by default, change LinearCombination.operation to SUM paramModulationOperation to ADD)
            if (self.mappingProjection.parameterStates and
                    self.receiver is self.mappingProjection.parameterStates[MATRIX]):
                self.receiver.paramsCurrent.update(weight_change_params)

            else:
                raise LearningSignalError("Receiver arg ({}) for {} must be the "
                                          "parameterStates[{}] param of the receiver".
                                          format(self.receiver, self.name, MATRIX))

        # Receiver was specified as a Mapping Projection
        elif isinstance(self.receiver, Mapping):

            self.mappingProjection = self.receiver

            from PsyNeuLink.Components.States.InputState import instantiate_state_list
            from PsyNeuLink.Components.States.InputState import instantiate_state

            # Check if Mapping Projection has parameterStates Ordered Dict and MATRIX entry
            try:
                self.receiver.parameterStates[MATRIX]

            # receiver does NOT have parameterStates attrib
            except AttributeError:
                # Instantiate parameterStates Ordered dict
                #     with ParameterState for receiver's functionParams[MATRIX] param
                self.receiver.parameterStates = instantiate_state_list(owner=self.receiver,
                                                                       state_list=[(MATRIX,
                                                                                    weight_change_params)],
                                                                       state_type=ParameterState,
                                                                       state_param_identifier=kwParameterState,
                                                                       constraint_value=self.mappingWeightMatrix,
                                                                       constraint_value_name=LEARNING_SIGNAL,
                                                                       context=context)
                self.receiver = self.receiver.parameterStates[MATRIX]

            # receiver has parameterStates but not (yet!) one for MATRIX, so instantiate it
            except KeyError:
                # Instantiate ParameterState for MATRIX
                self.receiver.parameterStates[MATRIX] = instantiate_state(owner=self.receiver,
                                                                            state_type=ParameterState,
                                                                            state_name=MATRIX,
                                                                            state_spec=kwParameterState,
                                                                            state_params=weight_change_params,
                                                                            constraint_value=self.mappingWeightMatrix,
                                                                            constraint_value_name=LEARNING_SIGNAL,
                                                                            context=context)

            # receiver has parameterState for MATRIX, so update its params with ones specified by LearningSignal
            else:
                # MODIFIED 8/13/16:
                # FIX: ?? SHOULD THIS USE assign_defaults:
                self.receiver.parameterStates[MATRIX].paramsCurrent.update(weight_change_params)

            # Assign self.receiver to parameterState used for weight matrix param
            self.receiver = self.receiver.parameterStates[MATRIX]

        # If it is not a ParameterState or Mapping Projection, raise exception
        else:
            raise LearningSignalError("Receiver arg ({}) for {} must be a Mapping projection or"
                                      " a MechanismParatemerState of one".format(self.receiver, self.name))

        if kwDeferredDefaultName in self.name:
            self.name = self.mappingProjection.name + ' ' + self.componentName
            # self.name = self.mappingProjection.name + \
            #             self.mappingProjection.parameterStates[MATRIX].name + \
            #             ' ' + self.componentName

        # Assign errorSource as the MappingProjection's receiver mechanism
        self.errorSource = self.mappingProjection.receiver.owner

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
                                                                                             FUNCTION_PARAMS,
                                                                                             MATRIX)
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

    def _instantiate_sender(self, context=None):
        # DOCUMENT: SEE UPDATE BELOW
        """Assign self.variable to MonitoringMechanism output or self.receiver.receiverErrorSignals

        Call this after _instantiate_receiver, as that is needed to determine the sender (i.e., source of errorSignal)

        If sender arg or PROJECTION_SENDER was specified, it has been assigned to self.sender
            and has been validated as a MonitoringMechanism, so:
            - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
            - assign its outputState.value as self.variable
        If sender was not specified (i.e., passed as MonitoringMechanism_Base specified in paramClassDefaults):
           if the owner of the Mapping projection projects to a MonitoringMechanism, then
               - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
               - assign its outputState.value as self.variable
           UPDATE: otherwise, if Mapping projection's receiver has an errorSignal attribute, use that as self.variable
               (e.g., "hidden units in a multilayered neural network, using BackPropagation Function)
           [TBI: otherwise, implement default MonitoringMechanism]
           otherwise, raise exception

FROM TODO:
#    - _instantiate_sender:
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

        # FIX: 8/7/16 XXX
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
            # Validate that it belongs to a MonitoringMechanism
            if not isinstance(self.sender.owner, MonitoringMechanism_Base):
                raise LearningSignalError("OutputState ({}) specified as sender for {} belongs to a {}"
                                          " rather than a MonitoringMechanism".
                                          format(self.sender.name,
                                                 self.name,
                                                 self.sender.owner.__class__.__name__))
            self.validate_error_signal(self.sender.value)

            # - assign MonitoringMechanism's outputState.value as self.variable
            # FIX: THIS DOESN"T SEEM TO HAPPEN HERE.  DOES IT HAPPEN LATER??

            # Add reference to MonitoringMechanism to Mapping projection
            monitoring_mechanism = self.sender

        # MonitoringMechanism class (i.e., not an instantiated object) specified for sender, so instantiate it:
        # - for terminal mechanism of Process, instantiate Comparator MonitoringMechanism
        # - for preceding mechanisms, instantiate WeightedSum MonitoringMechanism
        else:
            # Get errorSource:  ProcessingMechanism for which error is being monitored
            #    (i.e., the mechanism to which the Mapping projection projects)
            # Note: Mapping._instantiate_receiver has not yet been called, so need to do parse below
            from PsyNeuLink.Components.States.InputState import InputState
            if isinstance(self.mappingProjection.receiver, Mechanism):
                self.errorSource = self.mappingProjection.receiver
            elif isinstance(self.mappingProjection.receiver, InputState):
                self.errorSource = self.mappingProjection.receiver.owner

            next_level_monitoring_mechanism = None

            # Check if errorSource has a projection to a MonitoringMechanism or a ProcessingMechanism
            for projection in self.errorSource.outputState.sendsToProjections:
                # errorSource has a projection to a MonitoringMechanism, so validate it, assign it and quit search
                if isinstance(projection.receiver.owner, MonitoringMechanism_Base):
                    self.validate_error_signal(projection.receiver.owner.outputState.value)
                    monitoring_mechanism = projection.receiver.owner
                    break
                # errorSource has a projection to a ProcessingMechanism, so:
                #   - determine whether that has a LearningSignal
                #   - if so, get its MonitoringMechanism and weight matrix (needed by BackProp)
                if isinstance(projection.receiver.owner, ProcessingMechanism_Base):
                    try:
                        next_level_learning_signal = projection.parameterStates[MATRIX].receivesFromProjections[0]
                    except (AttributeError, KeyError):
                        # Next level's projection has no parameterStates, Matrix parameterState or projections to it
                        #    => no LearningSignal
                        pass # FIX: xxx ?? ADD LearningSignal here if requested?? or intercept error message to do so?
                    else:
                        # Next level's projection has a LearningSignal so get:
                        #     the weight matrix for the next level's projection
                        #     the MonitoringMechanism that provides error_signal
                        # next_level_weight_matrix = projection.matrix
                        next_level_monitoring_mechanism = next_level_learning_signal.sender

            # errorSource does not project to a MonitoringMechanism
            if not monitoring_mechanism:

                # FIX:  NEED TO DEAL WITH THIS RE: RL -> DON'T CREATE BACK PROJECTIONS??
                # NON-TERMINAL Mechanism
                # errorSource at next level projects to a MonitoringMechanism:
                #    instantiate WeightedError MonitoringMechanism and the back-projection for its error signal:
                #        (computes contribution of each element in errorSource to error at level to which it projects)
                if next_level_monitoring_mechanism:
                    error_signal = np.zeros_like(next_level_monitoring_mechanism.value)
                    monitoring_mechanism = WeightedError(error_signal=error_signal,
                                                         params={NEXT_LEVEL_PROJECTION:projection},
                                                         name=self.mappingProjection.name + " Weighted_Error")

                    # Instantiate mapping projection to provide monitoring_mechanism with error signal
                    Mapping(sender=next_level_monitoring_mechanism,
                            receiver=monitoring_mechanism,
                            # name=monitoring_mechanism.name+'_'+MAPPING)
                            matrix=IDENTITY_MATRIX,
                            name=next_level_monitoring_mechanism.name +
                                 ' to '+monitoring_mechanism.name +
                                 ' ' + MAPPING + ' Projection')

                # TERMINAL Mechanism
                # errorSource at next level does NOT project to a MonitoringMechanism:
                #     instantiate DefaultTrainingMechanism MonitoringMechanism
                #         (compares errorSource output with external training signal)
                else:
                    # # MODIFIED 9/4/16 OLD:
                    # output_signal = np.zeros_like(self.errorSource.outputState.value)
                    # MODIFIED 9/4/16 NEW:
                    if self.function.componentName is kwBackProp:
                        output_signal = np.zeros_like(self.errorSource.outputState.value)
                    # Force smaple and target of Comparartor to be scalars for RL
                    elif self.function.componentName is kwRL:
                        output_signal = np.array([0])
                    else:
                        raise LearningSignalError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                                  format(self.function.name, self.name))
                    # MODIFIED 9/4/16 END
                    # IMPLEMENTATION NOTE: training_signal assignment currently assumes training mech is Comparator
                    training_signal = output_signal
                    training_mechanism_input = np.array([output_signal, training_signal])
                    monitoring_mechanism = DefaultTrainingMechanism(training_mechanism_input)
                    # Instantiate a mapping projection from the errorSource to the DefaultTrainingMechanism
                    try:
                        monitored_state = self.errorSource.paramsCurrent[MONITOR_FOR_LEARNING]
                        monitored_state = self.errorSource.outputStates[monitored_state]
                    except KeyError:
                        # No state specified so use Mechanism as sender arg
                        monitored_state = self.errorSource
                    if self.function.componentName is kwBackProp:
                        matrix = IDENTITY_MATRIX
                    # Force sample and target of Comparator to be scalars for RL
                    elif self.function.componentName is kwRL:
                        matrix = FULL_CONNECTIVITY_MATRIX
                    self.monitoring_projection = Mapping(sender=monitored_state,
                                                         receiver=monitoring_mechanism.inputStates[COMPARATOR_SAMPLE],
                                                         name=self.errorSource.name +
                                                              ' to '+
                                                              monitoring_mechanism.name+' ' +
                                                              MAPPING+' Projection',
                                                         matrix=matrix)

            self.sender = monitoring_mechanism.outputState

            # "Cast" self.variable to match value of sender (MonitoringMechanism) to pass validation in add_to()
            # Note: self.variable will be re-assigned in _instantiate_function()
            self.variable = self.errorSignal

            # Add self as outgoing projection from MonitoringMechanism
            from PsyNeuLink.Components.Projections.Projection import _add_projection_from
            _add_projection_from(sender=monitoring_mechanism,
                                state=monitoring_mechanism.outputState,
                                projection_spec=self,
                                receiver=self.receiver,
                                context=context)

        # VALIDATE THAT OUTPUT OF SENDER IS SAME LENGTH AS THIRD ITEM (ERROR SIGNAL) OF SEL.FFUNCTION.VARIABLE

        # Add reference to MonitoringMechanism to Mapping projection
        self.mappingProjection.monitoringMechanism = monitoring_mechanism

    def validate_error_signal(self, error_signal):
        """Check that error signal (MonitoringMechanism.outputState.value) conforms to what is needed by self.function
        """

        if self.function.componentName is kwRL:
            # The length of the sender (MonitoringMechanism)'s outputState.value (the error signal) must == 1
            #     (since error signal is a scalar for RL)
            if len(error_signal) != 1:
                raise LearningSignalError("Length of error signal ({}) received by {} from {}"
                                          " must be 1 since {} uses {} as its learning function".
                                          format(len(error_signal), self.name, self.sender.owner.name, self.name, kwRL))
        if self.function.componentName is kwBackProp:
            # The length of the sender (MonitoringMechanism)'s outputState.value (the error signal) must be the
            #     same as the width (# columns) of the Mapping projection's weight matrix (# of receivers)
            if len(error_signal) != self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]:
                raise LearningSignalError("Length of error signal ({}) received by {} from {} must match the"
                                          "receiver dimension ({}) of the weight matrix for {}".
                                          format(len(error_signal),
                                                 self.name,
                                                 self.sender.owner.name,
                                                 len(self.mappingWeightMatrix.shape[WT_MATRIX_RECEIVERS_DIM]),
                                                 self.mappingProjection))
        else:
            raise LearningSignalError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                      format(self.function.name, self.name))



    def _instantiate_function(self, context=None):
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

        super()._instantiate_function(context)

        from PsyNeuLink.Components.Functions.Function import ACTIVATION_FUNCTION
        # Insure that the learning function is compatible with the activation function of the errorSource
        error_source_activation_function_type = type(self.errorSource.function_object)
        function_spec = self.function_object.paramsCurrent[ACTIVATION_FUNCTION]
        if isinstance(function_spec, TransferFunction):
            learning_function_activation_function_type = type(function_spec)
        elif issubclass(function_spec, TransferFunction):
            learning_function_activation_function_type = function_spec
        else:
            raise LearningSignalError("PROGRAM ERROR: activation function ({}) for {} is not a TransferFunction".
                                      format(function_spec, self.name))
        if error_source_activation_function_type != learning_function_activation_function_type:
            raise LearningSignalError("Activation function ({}) of error source ({}) is not compatible with "
                                      "the activation function ({}) specified for {}'s function ({}) ".
                                      format(error_source_activation_function_type.__name__,
                                             self.errorSource.name,
                                             learning_function_activation_function_type.__name__,
                                             self.name,
                                             self.params[FUNCTION].__self__.__class__.__name__))

        # FIX: MOVE TO AFTER INSTANTIATE FUNCTION??
        # IMPLEMENTATION NOTE:  MOVED FROM _instantiate_receiver
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

    # # MODIFIED 9/4/16 OLD:
    # def execute(self, input=NotImplemented, params=NotImplemented, time_scale=None, context=None):
    # MODIFIED 9/4/16 NEW:
    def execute(self, input=NotImplemented, params=None, time_scale=None, context=None):
    # MODIFIED 9/4/16 END
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

        LearningSignal function:
            Generalized delta rule:
            weight = weight + (learningRate * errorDerivative * transferDerivative * sampleSender)
            for sumSquared error function:  errorDerivative = (target - sample)
            for logistic activation function: transferDerivative = sample * (1-sample)
        NEEDS:
        - errorDerivative:  get from FUNCTION of Comparator Mechanism
        - transferDerivative:  get from FUNCTION of Processing Mechanism

        :return: (2D np.array) self.weightChangeMatrix
        """
        # Pass during initialization (since has not yet been fully initialized
        if self.value is DEFERRED_INITIALIZATION:
            return self.value

        # GET INPUT TO Projection to Error Source:
        # Array of input values from Mapping projection's sender mechanism's outputState
        input = self.mappingProjection.sender.value

        # ASSIGN OUTPUT TO ERROR SOURCE
        # Array of output values for Mapping projection's receiver mechanism
        # output = self.mappingProjection.receiver.owner.outputState.value
# FIX: IMPLEMENT self.unconvertedOutput AND self.convertedOutput, VALIDATE QUANTITY BELOW IN _instantiate_sender, 
# FIX:   ASSIGN self.input ACCORDINGLY
        output = self.errorSource.outputState.value

        # ASSIGN ERROR
# FIX: IMPLEMENT self.input AND self.convertedInput, VALIDATE QUANTITY BELOW IN _instantiate_sender, ASSIGN ACCORDINGLY
        error_signal = self.errorSignal

        # CALL function TO GET WEIGHT CHANGES
        # rows:  sender errors;  columns:  receiver errors
        self.weightChangeMatrix = self.function([input, output, error_signal], params=params, context=context)

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.weightChangeMatrix))

        self.value = self.weightChangeMatrix

        return self.value

    @property
    def errorSignal(self):
        return self.sender.value
