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

A LearningProjection takes as its input the output of its `sender <LearningProjection.sender>` --
a `MonitoringMechanism <MonitoringMechanism>`.  It uses this as an `error_signal` to compute a `weight_change_matrix`
that is assigned as the `value <LearningProjection.value>` of the LearningProjection.  This is used, in turn,
by the LearningProjection's `receiver <LearningProjection.receiver>` -- a `parameterState <ParameterState>` for
a `MappingProjection` to modify its  `matrix <MappingProjection.matrix>` parameter.  A LearningProjection can
be assigned different `functions <LearningProjection_Function>` to implement different learning algorithms,
which are associated with corresponding types of `MonitoringMechanisms <MonitoringMechanism>`.


.. _LearningProjection_Creation:

Creating a LearningProjection
------------------------------------

COMMENT:
    NOTE HERE THAT EITHER SENDER OR RECEIVER HAS TO BE SPECIFIED FOR FINAL IMPLEMENTATION
    IF IT IS A RECEIVER, THEN APPRPRIATE SENDER COMPONENTS (LearningMechanism and, if necessary ObjectiveMechanism)
        WILL BE INSTANTIATED
    IF IT IS A SENDER, THEN ?? HOW IS RECEIVER IDENTIFIED??
COMMENT

.. _LearningProjection_Automatic_Creation:

Automatic creation
~~~~~~~~~~~~~~~~~~

.. _LearningProjection_Simple_Learning_Figure:


.. _LearningProjection_Structure:

Structure
---------


.. _LearningProjection_MonitoringMechanism:

.. _LearningProjection_Targets:

.. _LearningProjection_Target_vs_Terminal_Figure:

.. _LearningProjection_Function:

.. _LearningProjection_Execution:

Execution
---------

.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""


from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.ComparatorMechanism import ComparatorMechanism, \
                                                                                      COMPARATOR_SAMPLE
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import MonitoringMechanism_Base
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.WeightedErrorMechanism import WeightedErrorMechanism, \
                                                                                         PROJECTION_TO_NEXT_MECHANISM
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import _objective_mechanism_role
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.Functions.Function import BackPropagation, Logistic
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism import LearningMechanism

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


WEIGHT_CHANGE_PARAMS = "weight_change_params"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

TARGET_ERROR = "TARGET_ERROR"
TARGET_ERROR_MEAN = "TARGET_ERROR_MEAN"
TARGET_ERROR_SUM = "TARGET_ERROR_SUM"
TARGET_SSE = "TARGET_SSE"
TARGET_MSE = "TARGET_MSE"


DefaultTrainingMechanism = ObjectiveMechanism

class LearningProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LearningProjection(Projection_Base):
    """
    LearningProjection(               \
                 sender=None,         \
                 receiver=None,       \
                 learning_rate=None,  \
                 params=None,         \
                 name=None,           \
                 prefs=None)

    Implements a projection that modifies the matrix parameter of a MappingProjection.

    COMMENT:
        Description:
            The LearningProjection class is a componentType in the Projection category of Function.
            It implements a projection to the parameterState of a MappingProjection that modifies its matrix parameter.
            It's function takes the output of a LearningMechanism (its learning_signal attribute), and provides this
            to the parameterState to which it projects.

        Class attributes:
            + className = LEARNING_PROJECTION
            + componentType = PROJECTION
            + paramClassDefaults (dict) :
                default
                + FUNCTION (Function): default Linear
                + FUNCTION_PARAMS (dict):
                    + SLOPE (value) : default 1
                    + INTERCEPT (value) : default 0
                + WEIGHT_CHANGE_PARAMS (dict) :  # Determine how weight changes are applied to weight matrix
                                                 # Note:  assumes MappingProjection.function is LinearCombination
                    default
                    + FUNCTION_PARAMS: {OPERATION: SUM},
                    + PARAMETER_MODULATION_OPERATION: ModulationOperation.ADD,
                    + PROJECTION_TYPE: LEARNING_PROJECTION

            + paramNames (dict)
            + classPreference (PreferenceSet): LearningProjectionPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            function (computes function specified in params[FUNCTION]
    COMMENT

    Arguments
    ---------
    sender : Optional[MonitoringMechanism or OutputState of one]
        the source of the `error_signal` for the LearningProjection. If it is not specified, one will be
        `automatically created <LearningProjection_Automatic_Creation>` that is appropriate for the
        LearningProjection's `errorSource <LearningProjection.errorSource>`.

    receiver : Optional[MappingProjection or ParameterState for ``matrix`` parameter of one]
        the `parameterState <ParameterState>` (or the `MappingProjection` that owns it) for the
        `matrix <MappingProjection.MappingProjection.matrix>` to be modified by the LearningProjection.

    learning_rate : Optional[float]
        specifies a learning_rate for the LearningProjection, that will supercede any specified for the
        `LearningMechanism` from which it projects, of any `process <Process>` and/or `system <System>` to which that
        belongs (see `learning_rate <LearningProjection>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
        projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the projection's default `function <LearningProjection.function>` and parameter assignments.  Values specified
        for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default LearningProjection-<index>
        a string used for the name of the LearningProjection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Projection.classPreferences]
        the `PreferenceSet` for the LearningProjection.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    componentType : LEARNING_PROJECTION

    sender : OutputState of MonitoringMechanism
        source of `error_signal`.

    receiver : ParameterState of MappingProjection
        parameterState for the `matrix <MappingProjection.MappingProjection.matrix>` parameter of
        the `mappingProjection` to be modified by the LearningProjection.

    mappingProjection : MappingProjection
        the `MappingProjection` that owns the `parameterState <ParameterState>` to which the
        LearningProjection projects (i.e., that owns its `receiver <LearningProjection.receiver>`).

    variable : 2d np.array
        COMMENT:
            WRONG?  CORRECTED BELOW
            same as :py:data:`mappingProjection <LearningProjection.mappingProjection>`.
        COMMENT
        same as `learning_signal`.

    learning_signal : 2d np.array
        matrix of parameter (weight) changes to convey to the matrix parameter of the LearningProjection's
        `receiver <LearningProject.receiver>` (rows correspond to sender, columns to receiver).

    function : Function : default Linear
        assigns the learning_signal received from `LearningMechanism` as the value of the projection.

    weight_change_params : dict : default: see below
        specifies to `receiver <LearningProjection.receiver>` how the weight changes specified by the
        `learning_signal` received from the LearningMechanism should be applied to its matrix parameter.
        It assumes that the MappingProjection.function of the receiver is a LinearCombination Function.
        By default it includes the following entries:  `FUNCTION_PARAMS`: `OPERATION`: `SUM`,
        `PARAMETER_MODULATION_OPERATION`: `ModulationOperation.ADD`, `PROJECTION_TYPE`: `LEARNING_PROJECTION`.

    learning_rate : Optional[float]
        determines the learning_rate for the LearningProjection.  If specified, it supercedes any specified for
        the `LearningMechanism` from which it projects, of any `process <Process>` and/or `system <System>` to which that
        belongs (see `learning_rate <LearningProjection>` for details).  If is `None`, the learning_rate for the
        `LearningMechanism` will be used.

    weight_change_matrix : 2d np.array
        matrix of changes to be made to the `mappingWeightMatrix` (rows correspond to sender, columns to receiver);
        same as `value <LearningProjection.value>`.

    value : 2d np.array
        same as `weight_change_matrix`.

    name : str : default LearningProjection-<index>
        the name of the LearningProjection.
        Specified in the `name` argument of the constructor for the projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for projection.
        Specified in the `prefs` argument of the constructor for the projection;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    """

    componentType = LEARNING_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    variableClassDefault = None

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_SENDER: LearningMechanism,
                               PARAMETER_STATES: None, # This suppresses parameterStates
                               FUNCTION: Linear,
                               FUNCTION_PARAMS:
                                   {SLOPE: 1,
                                    INTERCEPT: 0},
                               WEIGHT_CHANGE_PARAMS:  # Determine how weight changes are applied to weight matrix
                                   {                  # Note:  assumes MappingProjection.function is LinearCombination
                                       FUNCTION_PARAMS: {OPERATION: SUM},
                                       PARAMETER_MODULATION_OPERATION: ModulationOperation.ADD,
                                       PROJECTION_TYPE: LEARNING_PROJECTION}
                               })

    @tc.typecheck
    def __init__(self,
                 sender:tc.optional(tc.any(OutputState, LearningMechanism))=None,
                 receiver:tc.optional(tc.any(ParameterState, MappingProjection))=None,
                 learning_rate:tc.optional(float)=None,
                 learning_function:tc.optional(is_function_type)=BackPropagation,
                 params:tc.optional(dict)=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(learning_function=learning_function,
                                                  learning_rate=learning_rate,
                                                  params=params)

        # Store args for deferred initialization
        self.init_args = locals().copy()
        self.init_args['context'] = self
        self.init_args['name'] = name
        del self.init_args['learning_function']
        del self.init_args['learning_rate']

        # Flag for deferred initialization
        self.value = DEFERRED_INITIALIZATION

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate sender and receiver

        Insure `sender <LearningProjection>` is an ObjectiveMechanism or the outputState of one.
        Insure `receiver <LearningProjection>` is a MappingProjection or the matrix parameterState of one.
        """

        # IMPLEMENTATION NOTE: IS TYPE CHECKING HERE REDUNDANT WITH typecheck IN __init__??

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # VALIDATE SENDER
        sender = self.sender
        if isinstance(sender, LearningMechanism):
            sender = self.sender = sender.outputState

        if any(s in {OutputState, LearningMechanism} for s in {sender, type(sender)}):
            # If it is the outputState of a MonitoringMechanism, check that it is a list or 1D np.array
            if isinstance(sender, OutputState):
                if not isinstance(sender.value, (list, np.ndarray)):
                    raise LearningProjectionError("Sender for {} (outputState of LearningMechanism {}) "
                                                  "must be a list or 1D np.array".format(self.name, sender.name))
                if not np.array(sender.value).ndim == 1:
                    raise LearningProjectionError("OutputState of {} (LearningMechanism for {})"
                                                  " must be an 1D np.array".format(sender.name, self.name))
            # If specification is a LearningMechanism class, pass (it will be instantiated in _instantiate_sender)
            elif inspect.isclass(sender) and issubclass(sender,  LearningMechanism):
                pass

        else:
            raise LearningProjectionError("The sender arg for {} ({}) must be an LearningMechanism, "
                                          "the outputState of one, or a reference to the class"
                                          .format(self.name, sender.name))


        # VALIDATE RECEIVER
        receiver = self.receiver
        if isinstance(receiver, MappingProjection):
            try:
                receiver = self.receiver = receiver.parameterStates[MATRIX]
            except KeyError:
                raise LearningProjectionError("The MappingProjection {} specified as the receiver for {} "
                                              "has no MATRIX parameter state".format(receiver.name, self.name))
        if not any(s in {ParameterState, MappingProjection} for s in {receiver, type(receiver)}):
            raise LearningProjectionError("The receiver arg for {} must be a MappingProjection "
                                          "or the MATRIX parameterState of one."
                                          .format(PROJECTION_SENDER, sender, self.name, ))

        # VALIDATE WEIGHT CHANGE PARAMS
        try:
            weight_change_params = target_set[WEIGHT_CHANGE_PARAMS]
        except KeyError:
            pass
        else:
            # FIX: CHECK THAT EACH ONE INCLUDED IS A PARAM OF A LINEAR COMBINATION FUNCTION
            for param_name, param_value in weight_change_params.items():
                if param_name is FUNCTION:
                    raise LearningProjectionError("{} of {} contains a function specification ({}) that would override "
                                                  "the LinearCombination function of the targeted MappingProjection".
                                                  format(WEIGHT_CHANGE_PARAMS,self.name,param_value))

    def _instantiate_sender(self, context=None):
        """Instantiate LearningMechanism
        """

        # LearningMechanism was not specified or it was specified by class,
        #    so call composition for "automatic" instantiation of a LearningMechanism
        # Note: this also instantiates an ObjectiveMechanism if necessary and assigns it the necessary projections
        if not isinstance(self.sender, (OutputState, LearningMechanism)):
            from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningAuxilliary \
                import _instantiate_learning_components
            _instantiate_learning_components(learning_projection=self,
                                             context=context + " " + self.name)

        if isinstance(self.sender, OutputState) and not isinstance(self.sender.owner, LearningMechanism):
            raise LearningProjectionError("Sender specified for LearningProjection {} ({}) is not a LearningMechanism".
                                          format(self.name, self.sender.owner.name))

        # This assigns self as an outgoing projection from the sender (LearningMechanism) outputState
        #    and formats self.variable to be compatible with that outputState's value (i.e., its learning_signal)
        super()._instantiate_sender(context=context)


    def _instantiate_receiver(self, context=None):
        """Validate that receiver has been assigned and is compatiable with the output of function

        Notes:
        * _validate_params verifies that receiver is a parameterState for the matrix parameter of a MappingProjection.
        * _super()._instantiate_receiver verifies that the projection has not already been assigned to the receiver.

        """

        super()._instantiate_receiver(context=context)

        # Insure that the learning_signal is compatible format with the receiver's weight matrix
        if not iscompatible(self.value, self.receiver.variable):
            raise LearningProjectionError("The learning_signal of {} ({}) is not compatible with the matrix of "
                                          "the MappingProjection ({}) to which it is being assigned ({})".
                                          format(self.name,
                                                 self.value,
                                                 self.receiver.value,
                                                 self.receiver.owner.name))

        # Insure that learning_signal has the same shape as the receiver's weight matrix
        try:
            receiver_weight_matrix_shape = np.array(self.receiver.value).shape
        except TypeError:
            receiver_weight_matrix_shape = 1
        try:
            learning_signal_shape = np.array(self.value).shape
        except TypeError:
            learning_signal_shape = 1

        if receiver_weight_matrix_shape != learning_signal_shape:
            raise ProjectionError("Shape ({}) of learing_signal matrix for {} from {}"
                                  " must match shape of the weight matrix ({}) for the receiver {}".
                                  format(learning_signal_shape,
                                         self.name,
                                         self.sender.name,
                                         receiver_weight_matrix_shape,
                                         self.receiver.owner.name))

        self.receiver.owner.monitoringMechanism = objective_mechanism


    def execute(self, input=None, clock=CentralClock, time_scale=None, params={}, context=None):
        """
        :return: (2D np.array) self.weight_change_matrix
        """

        # Pass during initialization (since has not yet been fully initialized
        if self.value is DEFERRED_INITIALIZATION:
            return self.value

        # # FIX: WHY DOESN"T THIS WORK: [ASSIGNMENT OF LEARNING_RATE TO SLOPE OF LEARNING FUNCTION]
        # # FIX: HANDLE THIS AS runtime_param?? OR JUST USE learning_rate TO MODULATE WEIGHT CHANGE MATRIX DIRECTLY?
        # if self.learning_rate:
        #     params.update({SLOPE:self.learning_rate})
        if self.learning_rate:
            self.learning_signal *= self.learning_rate

        self.weight_change_matrix = self.function(variable=self.learning_signal,
                                                  params=params,
                                                  context=context)

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.weight_change_matrix))

        # # TEST PRINT
        # print("\nr### WEIGHT CHANGES FOR {} TRIAL {}:\n{}".format(self.name, CentralClock.trial, self.value))

        return self.value

    @property
    def learning_signal(self):
        return self.variable

    @property
    def weight_change_matrix(self):
        return self.value

    @weight_change_matrix.setter
    def weight_change_matrix(self,assignment):
        self.value = assignment
