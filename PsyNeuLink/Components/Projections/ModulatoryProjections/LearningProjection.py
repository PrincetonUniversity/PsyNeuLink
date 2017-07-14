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

A LearningProjection is a subclass of `ModulatoryProjection` that projects from a `LearningMechanism` to the
MATRIX `parameterState <ParameterState>` of a `MappingProjection` and modifies the value of the
`matrix <MappingProjection.matrix>` parameter of that MappingProjection.  All of the LearningProjections in a System,
along with its other `learning components <LearningMechanism>`, can be displayed using the System's `show_graph` method
with its **show_learning** argument assigned as :keyword:`True`.

.. _LearningProjection_Creation:

Creating a LearningProjection
------------------------------------

A LearningProjection can be created using any of the standard ways to `create a Projection <Projection_Creation>`,
or by including it in a tuple that specifies the `matrix <Mapping_Matrix_Specification>` parameter for a
`MappingProjection`.  LearningProjections are also created automatically, along with the other
`components required for learning <LearningMechanism_Learning_Configurations>`, when learning is specified for a
`Process <Process_Learning>` or a `System <System_Execution_Learning>`.

If a LearningProjection is created using its constructor on its own, the `receiver <ControlProjection.receiver>`
argument must be specified.  If it is included in the `matrix specification <Mapping_Matrix_Specification>` for a
`MappingProjection`, the *MATRIX* `ParameterState` for the MappingProjection will be assigned as the
LearningProjection's `receiver <LearningProjection.receiver>`.  If its `sender <LearningProjection.sender>` is not
specified, its assignment depends on the `receiver <LearningProjection.receiver>`.  If the receiver belongs to a
MappingProjection that projects between two mechanisms that are both in the same `Process <Process_Learning>` or
`System <System_Execution_Learning>`, then the LearningProjection's `sender <LearningProjection.sender>` is assigned
to a `LearningSignal` of the `LearningMechanism` for the MappingProjection. If there is none, it is
`created <LearningMechanism_Creation>` along with any other components
needed to implement learning for the MappingProjection (see `LearningMechanism_Learning_Configurations`).

.. _LearningProjection_Deferred_Initialization:

Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~

When a LearningProjection is created, its full initialization is :ref:`deferred <Component_Deferred_Init>` until its
`sender <LearningProjection.sender>` and `receiver <LearningProjection.receiver>` have been fully specified.  This
allows a LearningProjection to be created before its `sender` and/or `receiver` have been created (e.g., before them
in a script), by calling its constructor without specifying its **sender** or **receiver** arguments.
However, for the LearningProjection to be operational, initialization must be completed by calling its `deferred_init`
method.  This is not necessary if learning has been specified for a `System <System_Execution_Learning>`,
`Process <Process_Learning>`, or as the `Projection <MappingProjection_Tuple_Specification>` in the `pathway` of a
Process -- in those cases, deferred initialization is completed automatically.

.. _LearningProjection_Structure:

Structure
---------

The `sender <LearningProjection.sender>` of a LearningProjection is a `LearningSignal` of a `LearningMechanism`.  Its
`receiver <LearningProjection.receiver>` is the *MATRIX* `ParameterState` of a `MappingProjection`,  The
`function <LearningProjection.function>` of a LearningProjection is, by default, the identity function;  that is,
it conveys the `value <LearningSignal.value>` of its `sender <LearningProjection.sender>` to
its `receiver <LearningProjection.receiver>`, for use in modifying the `matrix <MappingProjection.matrix>` parameter
of the `MappingProjection` being learned.

.. _LearningProjection_Execution:

Execution
---------

A LearningProjection cannot be executed directly.  It is executed when the *MATRIX* ParameterState to which it
projects is updated.  This occurs when the `learned_projection <LearningProjection.learned_projection>` (i.e.,
`MappingProjection` to which the *MATRIX* ParameterState belongs) is updated. Note that these events occur only
when the ProcessingMechanism that receives the `learned_projection <LearningProjection.learned_projection>` is
executed (see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating). When the LearningProjection is
executed, it gets the `learning_signal <LearningProjection.learning_signal>` from its
`sender <LearningProjection.sender>` (i.e., the `value <LearningSignal.value>` of the `LearningSignal` from which it
projects), and conveys that to its `receiver <LearningProjection.receiver>`, possibly modified by its `learning_rate
<LearningProjection.learning_rate>` parameter if that is specified (or it is specified for its
`sender <LearningProjection.sender>`).

.. note::
   The changes in the `matrix <MappingProjection.maitrix>` parameter of a `MappingProjection` in response to the
   execution of a LearningProjection are not applied until the `Mechanism` that receives MappingProjection are next
   executed; see :ref:`Lazy Evaluation` for an explanation of "lazy" updating).


.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import Linear, BackPropagation
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism \
    import LearningMechanism, ERROR_SIGNAL
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism import \
    ObjectiveMechanism
from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection import ModulatoryProjection_Base
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState

# Params:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

def _is_learning_spec(spec):
    """Evaluate whether spec is a valid learning specification

    Return `True` if spec is LEARNING or a valid projection_spec (see Projection._is_projection_spec.
    Otherwise, return `False`

    """
    if spec in {LEARNING, ENABLED}:
        return True
    else:
        return _is_projection_spec(spec)


WEIGHT_CHANGE_PARAMS = "weight_change_params"

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

DefaultTrainingMechanism = ObjectiveMechanism

class LearningProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)



class LearningProjection(ModulatoryProjection_Base):
    """
    LearningProjection(               \
                 sender=None,         \
                 receiver=None,       \
                 learning_function,   \
                 learning_rate=None,  \
                 params=None,         \
                 name=None,           \
                 prefs=None)

    Implements a Projection that modifies the matrix parameter of a MappingProjection.

    COMMENT:
        Description:
            The LearningProjection class is a componentType in the Projection category of Function.
            It implements a Projection from the LEARNING_SIGNAL outputState of a LearningMechanism to the MATRIX
            parameterState of a MappingProjection that modifies its matrix parameter.
            It's function takes the output of a LearningMechanism (its learning_signal attribute), and provides this
            to the parameterState to which it projects, possibly scaled by the LearningProjection's learning_rate.

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
                    + PARAMETER_MODULATION_OPERATION: Modulation.ADD,
                    + PROJECTION_TYPE: LEARNING_PROJECTION

            + classPreference (PreferenceSet): LearningProjectionPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            None
    COMMENT

    Arguments
    ---------
    sender : Optional[LearningMechanism or LearningSignal]
        the source of the `learning_signal` for the LearningProjection;  If it is not specified, one will be
        `automatically created <LearningProjection_Automatic_Creation>` that is appropriate for the
        LearningProjection's `errorSource <LearningProjection.errorSource>`.

    receiver : Optional[MappingProjection or ParameterState for ``matrix`` parameter of one]
        the `parameterState <ParameterState>` (or the `MappingProjection` that owns it) for the
        `matrix <MappingProjection.MappingProjection.matrix>` to be modified by the LearningProjection.

    learning_function : Optional[LearningFunction or function] : default BackPropagation
        specifies a function to be used for learning by the `sender <LearningMechanism.sender>` (i.e., its
        `function <LearningMechanism.function>` attribute).

    learning_rate : Optional[float or int]
        if specified, it is applied multiplicatively to `learning_signal` received from the `LearningSignal`
        from which it projects (see `learning_rate <LearningProjection.learning_rate>` for additional details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
        Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default `function <LearningProjection.function>` and parameter assignments.  Values specified
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

    sender : LEARNING_SIGNAL OutputState of a LearningMechanism
        source of `learning_signal <LearningProjection.learning_signal>`.

    receiver : MATRIX ParameterState of a MappingProjection
        parameterState for the `matrix <MappingProjection.MappingProjection.matrix>` parameter of
        the `learned_projection` to be modified by the LearningProjection.

    learned_projection : MappingProjection
        the `MappingProjection` to which LearningProjection projects, and to which its
        `receiver <LearningProjection.receiver>` and the `matrix <MappingProjection>` parameter to be modified belong.

    variable : 2d np.array
        same as `learning_signal <LearningProjection.learning_signal>`.

    learning_signal : 2d np.array

        matrix of "weight" changes calculated by the LearningProjection's `sender <LearningProjection.sender>` and
        used to modify the `matrix <MappingProjection.matrix>` parameter of its `receiver <LearningProjection.receiver>`
        (i.e., the `learned_projection <LearningProjection.learned_projection>`). Rows correspond to sender,
        columns to receiver (i.e., the input and output of the MappingProjection, respectively).

    function : Function : default Linear
        assigns the `learning_signal` received from `LearningMechanism` as the value of the Projection.

    weight_change_params : dict : default: see below
       specifies to the `receiver <LearningProjection.receiver>` how the `weight_change_matrix` should be applied to the
       `matrix <MappingProjection.matrix> parameter of its `receiver <LearningProjection.receiver>`.  It assumes that the
       `function <MappingProjection.function>` of the receiver is `LinearCombination`.  By default it passes
       the following dictionary of specifications to the `receiver <LearningProjection.receiver>`::
           FUNCTION_PARAMS: {OPERATION: SUM,
                             PARAMETER_MODULATION_OPERATION: Modulation.ADD,
                             PROJECTION_TYPE: LEARNING_PROJECTION}

    learning_rate : Optional[float]
        determines the learning_rate for the LearningProjection.  If specified, it is applied multiplicatively to the
        `learning_signal <LearningProjection.learning_signal>` and thus can be used to modulate the learning rate in
        addition to (and on top of) one specified for the `LearningMechanism` or its
        `function <LearningMechanism.function>`.  Specification of the
        `learning_rate <LearningProjection.learning_rate>` for LearningProjection supercedes any specification(s) of
        the :keyword:`learning_rate` for any `Process <Process.Process_Base.learning_rate>` and/or
        `System <System.System_Base.learning_rate>` to which the LearningMechanism from which it projects belongs.
        (see `learning_rate <LearningMechanism_Learning_Rate>` of LearningMechanism for additional details).

    weight_change_matrix : 2d np.array
        matrix of changes to be made to the `mappingWeightMatrix`, after learning_rate has been applied to the
        `learning_signal <LearningProjection.learning_signal>`; same as `value <LearningProjection.value>`.

    value : 2d np.array
        same as `weight_change_matrix`.

    name : str : default LearningProjection-<index>
        the name of the LearningProjection.
        Specified in the **name** argument of the constructor for the Projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for Projection.
        Specified in the **prefs** argument of the constructor for the Projection;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
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
                                       PROJECTION_TYPE: LEARNING_PROJECTION}
                               })

    @tc.typecheck
    def __init__(self,
                 sender:tc.optional(tc.any(OutputState, LearningMechanism))=None,
                 receiver:tc.optional(tc.any(ParameterState, MappingProjection))=None,
                 learning_function:tc.optional(is_function_type)=BackPropagation,
                 learning_rate:tc.optional(tc.any(parameter_spec))=None,
                 params:tc.optional(dict)=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # IMPLEMENTATION NOTE:
        #     the learning_function argument is implemented to preserve the ability to pass a learning function
        #     specification from the specification of a LearningProjection (used to implement learning for a
        #     MappingProjection, e.g., in a tuple) to the LearningMechanism responsible for implementing the function

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

        if INITIALIZING in context:
            # VALIDATE SENDER
            sender = self.sender
            if isinstance(sender, LearningMechanism):
                if len(sender.learning_signals) > 1:
                    raise LearningProjectionError("PROGRAM ERROR: {} has more than one LearningSignal "
                                                  "which is not currently supported".format(sender.name))
                sender = self.sender = sender.learning_signals[0]

            if any(s in {OutputState, LearningMechanism} for s in {sender, type(sender)}):
                # If it is the outputState of a LearningMechanism, check that it is a list or 1D np.array
                if isinstance(sender, OutputState):
                    if not isinstance(sender.value, (list, np.ndarray)):
                        raise LearningProjectionError("Sender for \'{}\' (OutputState of LearningMechanism \'{}\') "
                                                      "must be a list or 1D np.array".format(self.name, sender.name))
                    if not np.array(sender.value).ndim == 1:
                        raise LearningProjectionError("OutputState of \'{}\' (LearningMechanism for \'{}\')"
                                                      " must be an 1D np.array".format(sender.owner.name, self.name))
                # If specification is a LearningMechanism class, pass (it will be instantiated in _instantiate_sender)
                elif inspect.isclass(sender) and issubclass(sender,  LearningMechanism):
                    pass

            else:
                raise LearningProjectionError("The sender arg for {} ({}) must be an LearningMechanism, "
                                              "the OutputState of one, or a reference to the class"
                                              .format(self.name, sender.name))


            # VALIDATE RECEIVER
            receiver = self.receiver
            if isinstance(receiver, MappingProjection):
                try:
                    receiver = self.receiver = receiver._parameter_states[MATRIX]
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
                        raise LearningProjectionError("{} of {} contains a function specification ({}) "
                                                      "that would override the LinearCombination function "
                                                      "of the targeted MappingProjection".
                                                      format(WEIGHT_CHANGE_PARAMS,self.name,param_value))

    def _instantiate_sender(self, context=None):
        """Instantiate LearningMechanism
        """

        # LearningMechanism was specified by class or was not specified,
        #    so call composition for "automatic" instantiation of a LearningMechanism
        # Note: this also instantiates an ObjectiveMechanism if necessary and assigns it the necessary projections
        if not isinstance(self.sender, (OutputState, LearningMechanism)):
            from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningAuxilliary \
                import _instantiate_learning_components
            _instantiate_learning_components(learning_projection=self,
                                             context=context + " " + self.name)

        if isinstance(self.sender, OutputState) and not isinstance(self.sender.owner, LearningMechanism):
            raise LearningProjectionError("Sender specified for LearningProjection {} ({}) is not a LearningMechanism".
                                          format(self.name, self.sender.owner.name))

        # This assigns self as an outgoing projection from the sender (LearningMechanism) outputState
        #    and formats self.variable to be compatible with that outputState's value (i.e., its learning_signal)
        super()._instantiate_sender(context=context)

        if self.sender.learning_rate is not None:
            self.learning_rate = self.sender.learning_rate

    def _instantiate_receiver(self, context=None):
        """Validate that receiver has been assigned and is compatible with the output of function

        Notes:
        * _validate_params verifies that receiver is a parameterState for the matrix parameter of a MappingProjection.
        * _super()._instantiate_receiver verifies that the projection has not already been assigned to the receiver.

        """

        super()._instantiate_receiver(context=context)

        # Insure that the learning_signal is compatible with the receiver's weight matrix
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


        # FIX: SHOULD TEST WHETHER IT CAN BE USED, NOT WHETHER IT IS THE SAME SHAPE
        # # MODIFIED 3/8/17 OLD:
        # if receiver_weight_matrix_shape != learning_signal_shape:
        #     raise ProjectionError("Shape ({}) of learing_signal matrix for {} from {}"
        #                           " must match shape of the weight matrix ({}) for the receiver {}".
        #                           format(learning_signal_shape,
        #                                  self.name,
        #                                  self.sender.name,
        #                                  receiver_weight_matrix_shape,
        #                                  self.receiver.owner.name))
        # MODIFIED 3/8/17 END

        learning_mechanism = self.sender.owner
        learned_projection = self.receiver.owner

        # Check if learning_mechanism receives a projection from an ObjectiveMechanism;
        #    if it does, assign it to the objective_mechanism attribute for the projection being learned
        candidate_objective_mech = learning_mechanism.input_states[ERROR_SIGNAL].path_afferents[0].sender.owner
        if isinstance(candidate_objective_mech, ObjectiveMechanism) and candidate_objective_mech._role is LEARNING:
            learned_projection.objective_mechanism = candidate_objective_mech
        learned_projection.learning_mechanism = learning_mechanism
        learned_projection.has_learning_projection = True


    def execute(self, input=None, clock=CentralClock, time_scale=None, params=None, context=None):
        """
        :return: (2D np.array) self.weight_change_matrix
        """

        params = params or {}

        # Pass during initialization (since has not yet been fully initialized
        if self.value is DEFERRED_INITIALIZATION:
            return self.value

        # if self.learning_rate:
        #     params.update({SLOPE:self.learning_rate})

        self.weight_change_matrix = self.function(variable=self.sender.value,
                                                  params=params,
                                                  context=context)

        if self.learning_rate is not None:
            self.weight_change_matrix *= self.learning_rate


        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.weight_change_matrix))

        # TEST PRINT
        # print("\n@@@ WEIGHT CHANGES FOR {} TRIAL {}:\n{}".format(self.name, CentralClock.trial, self.value))
        # print("\n@@@ WEIGHT CHANGES CALCULATED FOR {} TRIAL {}".format(self.name, CentralClock.trial))
        # TEST DEBUG MULTILAYER
        # print("\n{}\n@@@ WEIGHT CHANGES FOR {} TRIAL {}:\n{}".
        #       format(self.__class__.__name__.upper(), self.name, CentralClock.trial, self.value))

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
