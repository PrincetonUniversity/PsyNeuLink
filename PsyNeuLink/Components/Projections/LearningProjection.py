# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningProjection **********************************************************

"""
.. _LearningProjection_Overview:

Overview
--------

A LearningProjection takes a value (an *error signal*) from a :doc:`MonitoringMechanism` (its ``sender``),
and uses this to compute a :py:data:`weightChangeMatrix <LearningProjection.weightChangeMatrix>` that is assigned as
its value.  This is used to modify the :ref:`matrix <Mapping_Matrix>` parameter of a :doc:`MappingProjection`.  A
LearningProjection can be assigned different :ref:`functions <LearningProjection_Function> to implement different
learning algorithms, which are associated with corresponding types of :doc:`MonitoringMechanisms <MonitoringMechanism>`.

.. _LearningProjection_Creation:

Creating a LearningProjection
------------------------------------

A LearningProjection can be created in any of the ways that can be used to
:ref:`create a projection <Projection_Creation>`, or by including it in the specification of a
:ref:`system <System>`, :ref:`process <Process>`, or projection in the :ref:`pathway <_Process_Projections>`
of a process.  Its ``sender`` (the source of its error signal) must be a MonitoringMechanism, and its ``receiver``
must be the parameterState of a MappingProjection.  When a LearningProjection is created, its full initialization is
:ref:`deferred <Component_Deferred_Init>` until its ``sender`` and ``receiver`` have been fully specified.  This means
that it is possible to create a LearningProjection using its constructor without specifying either its ``sender`` or
its ``receiver``.

COMMENT:
   CURRENT
COMMENT
It is not necessary to assign a ``sender``;  if none is specified when it is initialized, a MonitoringMechanism of
the appropriate type will be created (see :ref:`Structure <LearningProjection_Structure>` below).  However,
a LearningProjection's ``receiver`` must be specified.  Once that is done, for the LearningProjection to be operational,
initializaton must be completed by calling its ``deferred_init`` method.  This is not necessary if the
LearningProjection is specified as part of a system, process, or a projection in the pathway of a process --
in those cases, deferred initialization is completed automatically.

COMMENT:
   REPLACE WITH THIS ONCE FUNCTIONALITY IS IMPLEMENTED
    Initialization will
    be  completed as soon as the LearningProjection has been assigned as the projection *from* a MonitoringMechanism (i.e.,
    it's ``sender`` has been specified) and as the projection *to* the parameterState of a MappingProjection (i.e.,
    its ``receiver`` has been specified).  This is handled automatically when the LearningProjection is specified as part of
    a system, process, or projection in the pathway of a process.
COMMENT

.. _LearningProjection_Automatic_Creation:

**Automatic creation**.  When learning is specified for a :ref:`system <System_Execution_Learning>`,
a :ref:`process <Process_Learning>`, or in a :ref:`tuple that specifies a projection <>`,   PsyNeuLink automatically
generates the LearningSignals, MonitoringMechanisms, and corresponding projections required for learning to occur.
More specifically, LearningSignals are automatically created for the relevant MappingProjection(s) (for a system or
process, this excludes input projections to their :keyword:`ORIGIN` mechansims and output projections to their
:keyword:`TERMINAL` mechanisms). Each LearningProjection is assigned the relevant
:py:data:`errorSource <LearningProjection.errorSource>` -- that is, the ProcessingMechanism that receives the
LearningProjection's :py:data:`mappingProjection <LearningProjection.mappingProjection>` (the one to which it projects
and modifies).  If the :py:data:`errorSource <LearningProjection.errorSource>` projects to a MonitoringMechanism,
then that is assigned as the LearningProjection's ``sender``.  If the :py:data:`errorSource
<LearningProjection.errorSource>` does not have a projection to a MonitoringMechanism, then one is created for it,
and a MappingProjection is created that projects to it from the :py:data:`errorSource <LearningProjection.errorSource>`.
The type of MonitoringMechanism created depends on where the  ProcessingMechanism sits in the processing stream.  If
it is a standalone mechanism, or the :keyword:`TERMINAL` mechanism of a process or system,
then a :doc:`ComparatorMechanism` mechanism is created (which compares the output of the
:py:data:`errorSource <LearningProjection.errorSource>` with a target to generate the
:py:data:`errorSignal <LearningProjection.errorSignal>`).  If the
:py:data:`errorSource <LearningProjection.errorSource>` projects to any other mechanisms,
then a :doc:`WeightedErrorMechanism` mechanism is created (which, in calculating the
:py:data:`errorSignal <LearningProjection.errorSignal>`, takes account of
the effect that the :py:data:`errorSource <LearningProjection.errorSource>` has on the output of the
ProcessingMechanism(s) to which it projects).


.. _LearningProjection_Structure:

Structure
---------

The components involved in learning are shown in :ref:`this figure <Process_Learning_Figure>`, and described below.

**MappingProjection**.  Learning Signals project to the parameterState for a
:py:data:`matrix <MappingProjection.MappingProjection.matrix>` parameter of a MappingProjection projection, stored in
the LearningProjection's :py:data:`mappingProjection <LearningProjection.mappingProjection>` attribute.  It modifies
the :py:data:`matrix <MappingProjection.MappingProjection.matrix>` based on the output of the ProcessingMechanism to
which the  :py:data:`mappingProjection <LearningProjection.mappingProjection>` projects.

**Error source**.  This is the ProcessingMechanism to which the
:py:data:`mappingProjection <LearningProjection.mappingProjection>` projects, and on which the *error signal*  is
based.  It projects to a MonitoringMechanism that calculates the error signal.  By default, the primary outputState
of the :py:data:`errorSource <LearningProjection.errorSource>` projects to the MonitoringMechanism, and its value is
used to calculate the :py:data:`errorSignal <LearningProjection.errorSignal>`. However, a different outputState can
be specified by including an entry with :keyword:`MONITOR_FOR_LEARNING` as its key in a params dictionary for the
:py:data:`errorSource <LearningProjection.errorSource>`, and assigning a list with the specified outputState(s) as
its value.

**Error Signal**.  This is computed by a MonitoringMechanism to which the error_source projects.  It compares the
output of the :py:data:`errorSource <LearningProjection.errorSource>` with a target value, and uses the disparity
to compute the error signal.  The MonitoringMechanism projects to the LearningProjection, serving at its ``sender``,
and the error signal is used by the LearningProjection as the ``variable`` for its ``function`` (it is also stored
in the :py:data:`errorSignal <LearningProjection.errorSignal>` attribute).

.. _LearningProjection_Function

**Function**.  This computes the changes to the :py:data:`matrix <MappingProjection.MappingProjection.matrix>` parameter
of the LearningProjection's :py:data:`mappingProjection <LearningProjection.mappingProjection>` required to reduce the
:py:data:`errorSignal <LearningProjection.errorSignal>` for its
:py:data:`errorSource <LearningProjection.errorSource>`.  The default is :py:class:`Function.BackPropagation` (also
known as the Generalized Delta Rule;
see `Rumelhart et al., 1986 <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_).
However, it can be assigned to other functions that implement different learning algorithms.  Assignment of the
LearningProjection's ``function`` must be appropriate for the ``function`` of its :py:data:`errorSource
<LearningProjection.errorSource>` (how the learning signal is computed depends on the nature of the function that
generated the error);  failure to match the ``function`` of the LearningProjection with the ``function`` of the
ProcessingMechanism that is its :py:data:`errorSource <LearningProjection.errorSource>` will generate an error.

**Target(s)**.  This specifies the value with which the output of an
:py:data:`errorSource <LearningProjection.errorSource>` is compared, to calculate the
:py:data:`errorSignal <LearningProjection.errorSignal>`.  It is required by the :doc:`run <Run>` method of a process
or sysetm when learning is specified. provides to the :keyword:`TARGET` inputState of a :doc:`ComparatorMechanism`
mechanism.

**Monitoring Mechanism**.  The :py:data:`errorSignal <LearningProjection.errorSignal>` itself is computed by a
:doc:`MonitoringMechanism` (the LearningProjection's ``sender``).  The type of MonitoringMechanism, and how it
computes its :py:data:`errorSignal <LearningProjection.errorSignal>`, depends on the
:py:data:`errorSource <LearningProjection.errorSource>`. If it is a :keyword:`TERMINAL` mechanism of a process or
system, the MonitoringMechanism is a :doc:`ComparatorMechanism` mechanism,  that compares the output of the
:py:data:`errorSource <LearningProjection.errorSource>` (which projects to the ComparatorMechanism's :keyword:`SAMPLE`
inputState) with a target value that is provided as input to the process or system when it is :ref:`run
<Run_Targets>` (and projects to the ComparatorMechanism's :keyword:`TARGET` inputState). If the
:py:data:`errorSource <LearningProjection.errorSource>` is not the final mechanism of a process or system,
then the MonitoringMechanism is a :doc:`WeightedErrorMechanism` mechanism, which uses the output of the
ProcessingMechanism(s) to which the :py:data:`errorSource <LearningProjection.errorSource>` projects to calculate the
:py:data:`errorSignal <LearningProjection.errorSignal>`.

.. _LearningProjection_Execution:

Execution
---------

LearningSignals are executed after all of the mechanisms in a process or system have executed, including the
MonitoringMechanisms that provide the errorSignal to each LearningProjection.  When the LearningProjection is
executed, it uses its :py:data:`errorSignal <LearningProjection.errorSignal>` to calculate changes to the
:py:data:`matrix <MapptingProjection.MappingProjection.matrix>` of its
:py:data:`mappingProjection <LearningProjection.mappingProjection>`. Changes to the matrix are calculated so as to
reduce the errorSignal. The changes are assigned as the ``value`` of the LearningProjection, but are not applied to
the matrix until the next time the :py:data:`mappingProjection <LearningProjection.mappingProjection>` is executed
(see :ref:`Lazy_Evaluation` for an explanation of "lazy" updating).

.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""


from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.ComparatorMechanism import ComparatorMechanism, SAMPLE
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import MonitoringMechanism_Base
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.WeightedErrorMechanism import WeightedErrorMechanism, NEXT_LEVEL_PROJECTION
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.Functions.Function import BackPropagation

# Params:

parameter_keywords.update({LEARNING_PROJECTION})
projection_keywords.update({LEARNING_PROJECTION})

def _is_learning_spec(spec):
    """Evaluate whether spec is a valid learning specification

    Return :keyword:`true` if spec is LEARNING or a valid projection_spec (see Projection._is_projection_spec
    Otherwise, return :keyword:`False`

    """
    if spec is LEARNING:
        return True
    else:
        return _is_projection_spec(spec)


kwWeightChangeParams = "weight_change_params"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

DefaultTrainingMechanism = ComparatorMechanism

class LearningProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningProjection(Projection_Base):
    """
    LearningProjection(                                                         \
                 sender=None,                                               \
                 receiver=None,                                             \
                 function=BackPropagation(learning_rate=1,                  \
                                          activation_function=Logistic()),  \
                 params=None,                                               \
                 name=None,                                                 \
                 prefs=None)


    Implements a projection that modifies the matrix param of a MappingProjection.

    COMMENT:
        Description:
            The LearningProjection class is a componentType in the Projection category of Function.
            It implements a projection to the parameterState of a MappingProjection that modifies its matrix parameter.
            It's execute method takes the output of a MonitoringMechanism (self.variable), and the input and output of
                the ProcessingMechanism to which its receiver MappingProjection projects, and generates a matrix of
                weight changes for the MappingProjection's matrix parameter.

        Class attributes:
            + className = LEARNING_PROJECTION
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
    sender : Optional[MonitoringMechanism or OutputState of one]
        the source of the :py:data:`mappingProjection <LearningProjection.mappingProjection>` for the
        LearningProjection. If it is not specified, one will be
        :ref:`automatically created <LearningProjection_Automatic_Creation>` that is appropriate for
        the LearningProjection's :py:data:`errorSource <LearningProjection.errorSource>`.

    receiver : Optional[MappingProjection or ParameterState for ``matrix`` parameter of one]
        a MappingProjection, the :py:data:`matrix <MappingProjection.MappingProjection.matrix>` of which is modified by
        the LearningProjection.

    params : Optional[Dict[param keyword, param value]]
        a dictionary that specifies the parameters for the projection, its function, and/or a custom function and its
        parameters (see :doc:`Component` for specification of a params dict). By default, it contains an entry for
        the projection's default ``function`` and parameter assignments.

    name : str : default LearningProjection-<index>
        a string used for the name of the LearningProjection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : Projection.classPreferences]
        the PreferenceSet for the LearningProjection.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].


    Attributes
    ----------

    sender : OutputState of MonitoringMechanism
        source of :py:data:`mappingProjection <LearningProjection.mappingProjection>`.

    receiver : ParameterState of MappingProjection
        parameterState for the :py:data:`matrix <MappingProjection.MappingProjection.matrix>` parameter of
        :py:data:`mappingProjection <LearningProjection.mappingProjection>` (the one to be modified by learning).

    mappingProjection : MappingProjection
        the MappingProjection to which the LearningProjection projects;  more specifically, the owner of the
        parameterState that is the LearningProjection's ``receiver``.

    mappingWeightMatrix : 2d np.array
        :py:data:`matrix <MappingProjection.MappingProjection.matrix>` parameter for the
        :py:data:`mappingProjection <LearningProjection.mappingProjection>`.

    errorSource : ProcessingMechanism
        mechanism to which :py:data:`mappingProjection <LearningProjection.mappingProjection>` projects.

    errorSignal : 1d np.array
        value used as the ``variable`` for the LearningProjection's ``function`` to determine changes to
        :py:data:`mappingWeightMatrix <LearningProjection.mappingWeightMatrix>`.

    variable : 1d np.array
        same as :py:data:`mappingProjection <LearningProjection.mappingProjection>`.

    weightChangeMatrix : 2d np.array
        matrix of changes to be made to the :py:data:`mappingWeightMatrix <LearningProjection.mappingWeightMatrix>`
        (rows correspond to sender, columns to receiver).

    value : 2d np.array
        same as :py:data:`weightChangeMatrix <LearningProjection.weightChangeMatrix>`.

    name : str : default LearningProjection-<index>
        the name of the LearningProjection.
        Specified in the name argument of the call to create the projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the PreferenceSet for projection.
        Specified in the prefs argument of the call to create the projection;
        if it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].


    """

    componentType = LEARNING_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_SENDER: MonitoringMechanism_Base,
                               PARAMETER_STATES: None, # This suppresses parameterStates
                               kwWeightChangeParams:  # Determine how weight changes are applied to weight matrix
                                   {                  # Note:  assumes MappingProjection.function is LinearCombination
                                       FUNCTION_PARAMS: {OPERATION: SUM},
                                       PARAMETER_MODULATION_OPERATION: ModulationOperation.ADD,
                                       PROJECTION_TYPE: LEARNING_PROJECTION}
                               })

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 function=BackPropagation(learning_rate=1,
                                          activation_function=Logistic),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function, params=params)

        # Store args for deferred initialization
        self.init_args = locals().copy()
        self.init_args['context'] = self
        self.init_args['name'] = name

        # Flag for deferred initialization
        self.value = DEFERRED_INITIALIZATION

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure sender is a MonitoringMechanism or ProcessingMechanism and receiver is a ParameterState or
        MappingProjection

        Validate send in params[PROJECTION_SENDER] or, if not specified, sender arg:
        - must be the outputState of a MonitoringMechanism (e.g., ComparatorMechanism or WeightedErrorMechanism)
        - must be a list or 1D np.array (i.e., the format of an errorSignal)

        Validate receiver in params[PARAMETER_STATES] or, if not specified, receiver arg:
        - must be either a MappingProjection or parameterStates[MATRIX]

         """

        # Parse params[PROJECTION_SENDER] if specified, and assign self.sender
        super()._validate_params(request_set, target_set, context)

        # VALIDATE SENDER
        sender = self.sender
        self._validate_sender(sender)

        # VALIDATE RECEIVER
        try:
            receiver = target_set[PARAMETER_STATES]
            self._validate_receiver(receiver)
        except (KeyError, LearningProjectionError):
            # PARAMETER_STATES not specified:
            receiver = self.receiver
            self._validate_receiver(receiver)

        # VALIDATE WEIGHT CHANGE PARAMS
        try:
            weight_change_params = target_set[kwWeightChangeParams]
        except KeyError:
            pass
        else:
            # FIX: CHECK THAT EACH ONE INCLUDED IS A PARAM OF A LINEAR COMBINATION FUNCTION
            for param_name, param_value in weight_change_params.items():
                if param_name is FUNCTION:
                    raise LearningProjectionError("{} of {} contains a function specification ({}) that would override"
                                              " the LinearCombination function of the targeted MappingProjection".
                                              format(kwWeightChangeParams,
                                                     self.name,
                                                     param_value))

    def _validate_sender(self, sender):
        """Make sure sender is a MonitoringMechanism or ProcessingMechanism or the outputState for one;
        """

        # If specified as a MonitoringMechanism, reassign to its outputState
        if isinstance(sender, MonitoringMechanism_Base):
            sender = self.sender = sender.outputState

        # If it is the outputState of a MonitoringMechanism, check that it is a list or 1D np.array
        if isinstance(sender, OutputState):
            if not isinstance(sender.value, (list, np.ndarray)):
                raise LearningProjectionError("Sender for {} (outputState of MonitoringMechanism {}) "
                                          "must be a list or 1D np.array".format(self.name, sender))
            if not np.array(sender.value).ndim == 1:
                raise LearningProjectionError("OutputState of MonitoringMechanism ({}) for {} must be an 1D np.array".
                                          format(sender, self.name))

        # If specification is a MonitoringMechanism class, pass (it will be instantiated in _instantiate_sender)
        elif inspect.isclass(sender) and issubclass(sender,  MonitoringMechanism_Base):
            pass

        else:
            raise LearningProjectionError("Sender arg (or {} param ({}) for must be a MonitoringMechanism, "
                                      "the outputState of one, or a reference to the class"
                                      .format(PROJECTION_SENDER, sender, self.name, ))

    def _validate_receiver(self, receiver):
        """Make sure receiver is a MappingProjection or the parameterState of one
        """

        if not isinstance(receiver, (MappingProjection, ParameterState)):
            raise LearningProjectionError("Receiver arg ({}) for {} must be a MappingProjection or a parameterState of one"
                                      .format(receiver, self.name))
        # If it is a parameterState and the receiver already has a parameterStates dict
        #     make sure the assignment is to its MATRIX entry
        if isinstance(receiver, ParameterState):
            if receiver.owner.parameterStates and not receiver is receiver.owner.parameterStates[MATRIX]:
                raise LearningProjectionError("Receiver arg ({}) for {} must be the {} parameterState of a"
                                          "MappingProjection".format(receiver, self.name, MATRIX, ))

        # Notes:
        # * if specified as a MappingProjection, it will be assigned to a parameter state in _instantiate_receiver
        # * the value of receiver will be validated in _instantiate_receiver

    def _instantiate_attributes_before_function(self, context=None):
        """Override super to call _instantiate_receiver before calling _instantiate_sender

        Call _instantiate_receiver first since both _instantiate_sender and _instantiate_function
            reference the MappingProjection's weight matrix: self.mappingProjection.matrix

        """
        # FIX: PROBLEM: _instantiate_receiver usually follows _instantiate_function,
        # FIX:          and uses self.value (output of function) to validate against receiver.variable

        self._instantiate_receiver(context)

        # # MODIFIED 8/14/16: COMMENTED OUT SINCE SOLVED BY MOVING add_to TO _instantiate_attributes_after_function
        # # "Cast" self.value to MappingProjection parameterState's variable to pass validation in _instantiate_sender
        # # Note: this is because _instantiate_sender calls _add_projection_to
        # # (since self.value is not assigned until _instantiate_function; it will be reassigned there)
        # self.value = self.receiver.variable

        super()._instantiate_attributes_before_function(context)

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

        weight_change_params = self.paramsCurrent[kwWeightChangeParams]

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
                                                 kwLinearCombination,
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
            #                                      kwLinear,
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
                                                                       state_param_identifier=kwParameterState,
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
                                                                            state_spec=kwParameterState,
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
        """Assign self.variable to MonitoringMechanism output or self.receiver.receiverErrorSignals

        Call this after _instantiate_receiver, as that is needed to determine the sender (i.e., source of errorSignal)

        If sender arg or PROJECTION_SENDER was specified, it has been assigned to self.sender
            and has been validated as a MonitoringMechanism, so:
            - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
            - assign its outputState.value as self.variable
        If sender was not specified (i.e., passed as MonitoringMechanism_Base specified in paramClassDefaults):
           if the owner of the MappingProjection projects to a MonitoringMechanism, then
               - validate that the length of its outputState.value is the same as the width (# columns) of MATRIX
               - assign its outputState.value as self.variable
           UPDATE: otherwise, if MappingProjection's receiver has an errorSignal attribute, use that as self.variable
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

        # FIX: 8/7/16 XXX
        # FIX: NEED TO DEAL HERE WITH CLASS SPECIFICATION OF MonitoringMechanism AS DEFAULT
        # FIX: OR HAVE ALREADY INSTANTIATED DEFAULT MONITORING MECHANISM BEFORE REACHING HERE
        # FIX: EMULATE HANDLING OF DefaultMechanism (for MappingProjection) AND DefaultController (for ControlProjection)

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
                raise LearningProjectionError("OutputState ({}) specified as sender for {} belongs to a {}"
                                          " rather than a MonitoringMechanism".
                                          format(self.sender.name,
                                                 self.name,
                                                 self.sender.owner.__class__.__name__))
            self._validate_error_signal(self.sender.value)

            # - assign MonitoringMechanism's outputState.value as self.variable
            # FIX: THIS DOESN"T SEEM TO HAPPEN HERE.  DOES IT HAPPEN LATER??

            # Add reference to MonitoringMechanism to MappingProjection
            monitoring_mechanism = self.sender

        # MonitoringMechanism class (i.e., not an instantiated object) specified for sender, so instantiate it:
        # - for terminal mechanism of Process, instantiate ComparatorMechanism MonitoringMechanism
        # - for preceding mechanisms, instantiate WeightedSum MonitoringMechanism
        else:
            # Get errorSource:  ProcessingMechanism for which error is being monitored
            #    (i.e., the mechanism to which the MappingProjection projects)
            # Note: MappingProjection._instantiate_receiver has not yet been called, so need to do parse below
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
                    self._validate_error_signal(projection.receiver.owner.outputState.value)
                    monitoring_mechanism = projection.receiver.owner
                    break
                # errorSource has a projection to a ProcessingMechanism, so:
                #   - determine whether that has a LearningProjection
                #   - if so, get its MonitoringMechanism and weight matrix (needed by BackProp)
                if isinstance(projection.receiver.owner, ProcessingMechanism_Base):
                    try:
                        next_level_LEARNING_PROJECTION = projection.parameterStates[MATRIX].receivesFromProjections[0]
                    except (AttributeError, KeyError):
                        # Next level's projection has no parameterStates, Matrix parameterState or projections to it
                        #    => no LearningProjection
                        pass # FIX: xxx ?? ADD LearningProjection here if requested?? or intercept error message to do so?
                    else:
                        # Next level's projection has a LearningProjection so get:
                        #     the weight matrix for the next level's projection
                        #     the MonitoringMechanism that provides error_signal
                        # next_level_weight_matrix = projection.matrix
                        next_level_monitoring_mechanism = next_level_LEARNING_PROJECTION.sender

            # errorSource does not project to a MonitoringMechanism
            if not monitoring_mechanism:

                # FIX:  NEED TO DEAL WITH THIS RE: RL -> DON'T CREATE BACK PROJECTIONS??
                # NON-TERMINAL Mechanism
                # errorSource at next level projects to a MonitoringMechanism:
                #    instantiate WeightedErrorMechanism MonitoringMechanism and the back-projection for its error signal:
                #        (computes contribution of each element in errorSource to error at level to which it projects)
                if next_level_monitoring_mechanism:
                    error_signal = np.zeros_like(next_level_monitoring_mechanism.value)
                    monitoring_mechanism = WeightedErrorMechanism(error_signal=error_signal,
                                                         params={NEXT_LEVEL_PROJECTION:projection},
                                                         name=self.mappingProjection.name + " Weighted_Error")

                    # Instantiate MappingProjection to provide monitoring_mechanism with error signal
                    MappingProjection(sender=next_level_monitoring_mechanism,
                            receiver=monitoring_mechanism,
                            # name=monitoring_mechanism.name+'_'+MAPPING_PROJECTION)
                            matrix=IDENTITY_MATRIX,
                            name=next_level_monitoring_mechanism.name +
                                 ' to '+monitoring_mechanism.name +
                                 ' ' + MAPPING_PROJECTION + ' Projection')

                # TERMINAL Mechanism
                # errorSource at next level does NOT project to a MonitoringMechanism:
                #     instantiate DefaultTrainingMechanism MonitoringMechanism
                #         (compares errorSource output with external training signal)
                else:
                    if self.function.componentName is kwBackProp:
                        output_signal = np.zeros_like(self.errorSource.outputState.value)
                    # Force smaple and target of Comparartor to be scalars for RL
                    elif self.function.componentName is kwRL:
                        output_signal = np.array([0])
                    else:
                        raise LearningProjectionError("PROGRAM ERROR: unrecognized learning function ({}) for {}".
                                                  format(self.function.name, self.name))
                    # IMPLEMENTATION NOTE: training_signal assignment currently assumes training mech is ComparatorMechanism
                    training_signal = output_signal
                    training_mechanism_input = np.array([output_signal, training_signal])
                    monitoring_mechanism = DefaultTrainingMechanism(training_mechanism_input)
                    # Instantiate a MappingProjection from the errorSource to the DefaultTrainingMechanism
                    try:
                        monitored_state = self.errorSource.paramsCurrent[MONITOR_FOR_LEARNING]
                        monitored_state = self.errorSource.outputStates[monitored_state]
                    except KeyError:
                        # No state specified so use Mechanism as sender arg
                        monitored_state = self.errorSource
                    if self.function.componentName is kwBackProp:
                        matrix = IDENTITY_MATRIX
                    # Force sample and target of ComparatorMechanism to be scalars for RL
                    elif self.function.componentName is kwRL:
                        matrix = FULL_CONNECTIVITY_MATRIX
                    self.monitoring_projection = MappingProjection(sender=monitored_state,
                                                         receiver=monitoring_mechanism.inputStates[SAMPLE],
                                                         name=self.errorSource.name +
                                                              ' to '+
                                                              monitoring_mechanism.name+' ' +
                                                              MAPPING_PROJECTION+' Projection',
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

        # Add reference to MonitoringMechanism to MappingProjection
        self.mappingProjection.monitoringMechanism = monitoring_mechanism

    def _validate_error_signal(self, error_signal):
        """Check that error signal (MonitoringMechanism.outputState.value) conforms to what is needed by self.function
        """

        if self.function.componentName is kwRL:
            # The length of the sender (MonitoringMechanism)'s outputState.value (the error signal) must == 1
            #     (since error signal is a scalar for RL)
            if len(error_signal) != 1:
                raise LearningProjectionError("Length of error signal ({}) received by {} from {}"
                                          " must be 1 since {} uses {} as its learning function".
                                          format(len(error_signal), self.name, self.sender.owner.name, self.name, kwRL))
        if self.function.componentName is kwBackProp:
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

    # # MODIFIED 9/4/16 OLD:
    # def execute(self, input=NotImplemented, params=NotImplemented, time_scale=None, context=None):
    # MODIFIED 9/4/16 NEW:
    def execute(self, input=None, params=None, time_scale=None, context=None):
    # MODIFIED 9/4/16 END
        """

        DOCUMENT:
        LearnningSignal (Projection):
            - sender:  output of Monitoring Mechanism
                default: receiver.owner.outputState.sendsToProjections.<MonitoringMechanism> if specified,
                         else default ComparatorMechanism
            - receiver: MappingProjection parameterState (or some equivalent thereof)

        MappingProjection should have LEARNING_PARAM which:
           - specifies LearningProjection
           - defaults to BP
        ?? - uses self.outputStates.sendsToProjections.<MonitoringMechanism> if specified

        LearningProjection function:
            Generalized delta rule:
            weight = weight + (learningRate * errorDerivative * transferDerivative * sampleSender)
            for sumSquared error function:  errorDerivative = (target - sample)
            for logistic activation function: transferDerivative = sample * (1-sample)
        NEEDS:
        - errorDerivative:  get from FUNCTION of ComparatorMechanism
        - transferDerivative:  get from FUNCTION of Processing Mechanism

        :return: (2D np.array) self.weightChangeMatrix
        """
        # Pass during initialization (since has not yet been fully initialized
        if self.value is DEFERRED_INITIALIZATION:
            return self.value

        # GET INPUT TO Projection to Error Source:
        # Array of input values from MappingProjection's sender mechanism's outputState
        input = self.mappingProjection.sender.value

        # ASSIGN OUTPUT TO ERROR SOURCE
        # Array of output values for MappingProjection's receiver mechanism
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
