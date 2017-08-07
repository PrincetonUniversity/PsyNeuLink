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
*MATRIX* `ParameterState` of a `MappingProjection`, and modifies the value of the `matrix <MappingProjection.matrix>`
parameter of that MappingProjection.  All of the LearningProjections in a System, along with its other `learning
components <LearningMechanism_Learning_Configurations>`, can be displayed using the System's `show_graph` method with
its **show_learning** argument assigned as `True`.

.. _LearningProjection_Creation:

Creating a LearningProjection
------------------------------------

A LearningProjection can be created using any of the standard ways to `create a Projection <Projection_Creation>`,
or by including it in a `tuple <MappingProjection_Tuple_Specification>` that specifies the `matrix
<MappingProjection.matrix>` parameter of a `MappingProjection`.  LearningProjections are also created automatically,
along with the other `Components required for learning <LearningMechanism_Learning_Configurations>`, when learning is
specified for a `Process <Process_Learning>` or a `System <System_Execution_Learning>`.

If a LearningProjection is created explicitly (using its constructor), and its **receiver** argument is not specified,
its initialization is `deferred <LearningProjection_Deferred_Initialization>`.  If it is included in the `matrix
specification <MappingProjection_Tuple_Specification>` for a `MappingProjection`, the *MATRIX* `ParameterState` for
the MappingProjection will be assigned as the LearningProjection's `receiver <LearningProjection.receiver>`.  If its
`sender <LearningProjection.sender>` is not specified, its assignment depends on the `receiver
<LearningProjection.receiver>`.  If the receiver belongs to a MappingProjection that projects between two Mechanisms
that are both in the same `Process <Process_Learning>` or `System <System_Execution_Learning>`, then the
LearningProjection's `sender <LearningProjection.sender>` is assigned to a `LearningSignal` of the `LearningMechanism`
for the MappingProjection. If there is none, it is `created <LearningMechanism_Creation>` along with any other
components needed to implement learning for the MappingProjection (see `LearningMechanism_Learning_Configurations`).
Otherwise, the LearningProjection's initialization is `deferred <LearningProjection_Deferred_Initialization>`.

.. _LearningProjection_Deferred_Initialization:

Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~

When a LearningProjection is created, its full initialization is `deferred <Component_Deferred_Init>` until its
`sender <LearningProjection.sender>` and `receiver <LearningProjection.receiver>` have been fully specified.  This
allows a LearningProjection to be created before its `sender` and/or `receiver` have been created (e.g., before them
in a script), by calling its constructor without specifying its **sender** or **receiver** arguments.
However, for the LearningProjection to be operational, initialization must be completed by calling its `deferred_init`
method.  This is not necessary if the LearningProjection is included in a `tuple specification
<MappingProjection_Tuple_Specification>` for the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`,
in which case deferred initialization is completed automatically when the `LearningMechanism` associated with that
MappingProjection is created for the `Process` or `System` to which it belongs (see `LearningMechanism_Creation`).

.. _LearningProjection_Structure:

Structure
---------

.. _LearningProjection_Sender:

Sender
~~~~~~

The `sender <LearningProjection.sender>` of a LearningProjection is a `LearningSignal` of a `LearningMechanism`,
The `value <LearningSignal.value>` of the `sender <LearningProjection.sender>` -- a matrix of weight changes --
is used by the LearningProjection as its `variable <LearningProjection.variable>`;  this is also assigned to its
`learning_signal <LearningProjection.learning_signal>` attribute, and serves as the input to the LearningProjection's
`function <LearningProjection.function>`.

.. _LearningProjection_Function_and_Learning_Rate:

Function and learning_rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default `function <LearningProjection.function>` of a LearningProjection is an identity function (`Linear` with
**slope**\\ =1 and **intercept**\\ =0).  However, its result can be modified by the LearningProjection's `learning_rate
<LearningProjection.learning_rate>` parameter (specified in the **learning_rate** argument of its constructor).
If specified, it is applied multiplicatively to the output of the LearningProjection's `function
<LearningProjection.function>`, and the result is assigned to the LearningProjection's `value
<LearningProjection.value>` and `weight_change_matrix <LearningProjection.weight_change_matrix>` attributes.  Thus,
the LearningProjection's `learning_rate <LearningProjection.learning_rate>` parameter can be used to modulate the the
`learning_signal <LearningProjection.learning_signal>` it receives, in addition to (and on top of) the effects of the
`learning_rate <LearningMechanism.learning_rate>` for the `LearningMechanism` from which it receives the
`learning_signal <LearningProjection.learning_signal>`. Specification of the `learning_rate
<LearningProjection.learning_rate>` for a LearningProjection supersedes any specification(s) of the
:keyword:`learning_rate` for any `Process <Process.Process_Base.learning_rate>` and/or `System
<System.System_Base.learning_rate>` to which the LearningMechanism from which it projects belongs (see `learning_rate
<LearningMechanism_Learning_Rate>` for additional details).  However, its `learning_rate
<LearningProjection.learning_rate>` can be specified by the `LearningSignal <LearningSignal_Learning_Rate>` that
is its `sender <LearningProjection.sender>`;  that specification takes precedence over the direct specification of
the `learning_rate <LearningProjection.learning_rate>` for the LearningProjection (i.e., in the **learning_rate**
argument of its constructor, or by direct assignment of a value to the attribute).  If a `learning_rate
<LearningProjection.learning_rate>` is not specified for the LearningProjection, then the result of its `function
<LearningProjection.function>` is assigned unmodified as the LearningProjection's `value <LearningProjection.value>`
(and `weight_change_matrix <LearningProjection.weight_change_matrix>` attributes.

.. _LearningProjection_Receiver:

Receiver
~~~~~~~~

The `receiver <LearningProjection.receiver>` of a LearningProject is the *MATRIX* `ParameterState` of a
`MappingProjection`, that uses the `weight_change_matrix <LearningProjection.weight_change_matrix>` provided by the
LearningProjection to modify the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` being
learned.

.. _LearningProjection_Execution:

Execution
---------

A LearningProjection cannot be executed directly.  It is executed when the *MATRIX* ParameterState to which it
projects is updated.  This occurs when the `learned_projection <LearningProjection.learned_projection>` (the
`MappingProjection` to which the *MATRIX* ParameterState belongs) is updated. Note that these events occur only
when the ProcessingMechanism that receives the `learned_projection <LearningProjection.learned_projection>` is
executed (see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating). When the LearningProjection is
executed, it gets the `learning_signal <LearningProjection.learning_signal>` from its
`sender <LearningProjection.sender>` and conveys that to its `receiver <LearningProjection.receiver>`,
possibly modified by a `learning_rate <LearningProjection.learning_rate>` if that is specified for it or its `sender
<LearningProjection.sender>` (see `above <LearningProjection_Function_and_Learning_Rate>`).

.. note::
   The changes to the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection` in response to the
   execution of a LearningProjection are not applied until the `Mechanism` that receives MappingProjection are
   next executed; see :ref:`Lazy Evaluation` for an explanation of "lazy" updating).


.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

import inspect

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import InitStatus, parameter_keywords
from PsyNeuLink.Components.Functions.Function import BackPropagation, Linear, is_function_type
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism \
    import LearningMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism import ERROR_SIGNAL, ObjectiveMechanism
from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection import ModulatoryProjection_Base
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection import Projection_Base, _is_projection_spec, projection_keywords
from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Globals.Keywords import ENABLED, FUNCTION, FUNCTION_PARAMS, INITIALIZING, INTERCEPT, LEARNING, LEARNING_PROJECTION, MATRIX, OPERATION, PARAMETER_STATES, PROJECTION_SENDER, PROJECTION_TYPE, SLOPE, SUM
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import iscompatible, parameter_spec
from PsyNeuLink.Scheduling.TimeScale import CentralClock

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

    Subclass of `ModulatoryProjection` that modulates the value of a `ParameterState` for the
    `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`.

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

            + classPreference (PreferenceSet): LearningProjectionPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE

        Class methods:
            None
    COMMENT

    Arguments
    ---------

    sender : Optional[LearningMechanism or LearningSignal]
        the source of the `learning_signal <LearningProjection.learning_signal>` for the LearningProjection;  If it is
        not specified, initialization will be `deferred <LearningProjection_Deferred_Initialization>`.

    receiver : Optional[MappingProjection or ParameterState for matrix parameter of one]
        the *MATRIX* `ParameterState` (or the `MappingProjection` that owns it) for the `matrix
        <MappingProjection.matrix>` of the `learned_projection <LearningProjection.learned_projection>` to be
        modified by the LearningProjection.

    learning_function : Optional[LearningFunction or function] : default BackPropagation
        specifies a function to be used for learning by the `LearningMechanism` to which the
        LearningProjection's `sender <LearningProjection.sender>` belongs.

        .. note::
           the **learning_function** argument is implemented to preserve backward compatibility with previous versions;
           its use is not advised.

    learning_rate : Optional[float or int]
        if specified, it is applied multiplicatively to the `learning_signal <LearningProjection.learning_signal>`
        received from the `sender <LearningProjection.sender>`; specification of the `learning_rate
        <LearningProjection.learning_rate>` for a LearningProjection supersedes any specification(s) of the
        :keyword:`learning_rate` for any `Process <Process.Process_Base.learning_rate>` and/or `System
        <System.System_Base.learning_rate>` to which the LearningProjection belongs, and is
        applied in addition to any effects of the `learning_rate <LearningMechanism.learning_rate>` for the
        `LearningMechanism` from which the LearningProjection receives its `learning_signal
        <LearningProjection.learning_signal>` (see `LearningProjection_Function_and_Learning_Rate` for additional
        details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
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

    sender : LearningSignal
        source of `learning_signal <LearningProjection.learning_signal>`
        (see `LearningProjection_Sender` for additional details).

    receiver : MATRIX ParameterState of a MappingProjection
        *MATRIX* `ParameterState` for the `matrix <MappingProjection.MappingProjection.matrix>` parameter of the
        `learned_projection <LearningProjection.learned_projection>` (see `LearningProjection_Receiver` for additional
        details).

    learned_projection : MappingProjection
        the `MappingProjection` to which LearningProjection's `receiver <LearningProjection.receiver>` belongs.

    variable : 2d np.array
        same as `learning_signal <LearningProjection.learning_signal>`.

    learning_signal : 2d np.array
        matrix of weight changes calculated by the `LearningMechanism` to which the LearningProjection's  `sender
        <LearningProjection.sender>` belongs; rows correspond to the `sender <MappingProjection.sender>` of the
        `learned_projection <LearningProjection>`, and columns to its `receiver <MappingProjection.receiver>`
        (i.e., the input and output of the `learned_projection <LearningProjection>`, respectively).

    function : Function : default Linear
        assigns the `learning_signal` received from the `receiver <LearningProjection.receiver>` to the
        LearningProjection's `value <LearningProjection.value>`, possibly modified by its `learning_rate
        <LearningProjection.learning_rate>`.

    learning_rate : Optional[float]
        determines the learning_rate for the LearningProjection.  If specified, it is applied multiplicatively to the
        `learning_signal <LearningProjection.learning_signal>`; its specification may be superseded by the
        `learning_rate <LearningSignal.learning_rate>` of its `sender <LearningProjection.sender>`
        (see `LearningProjection_Function_and_Learning_Rate` for additional details);

    weight_change_matrix : 2d np.array
        output of the LearningProjection's `function <LearningProjection.function>`, possibly modified by its
        `learning_rate <LearningProjection.learning_rate>`;  reflects the matrix of weight changes to be made to the
        `matrix <MappingProjection.matrix>` parameter of the `learned_projection
        <LearningProjection.learned_projection>` (rows correspond to the `sender <MappingProjection.sender>` of the
        `learned_projection <LearningProjection.learned_projection>`, columns to its `receiver
        <MappingProjection.receiver>` (i.e., the input and output of the `learned_projection
        <LearningProjection.learned_projection>`, respectively).

    value : 2d np.array
        same as `weight_change_matrix`.

    name : str : default LearningProjection-<index>
        the name of the LearningProjection.
        Specified in the **name** argument of the constructor for the LearningProjection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for LearningProjection.
        Specified in the **prefs** argument of the constructor for the LearningProjection;
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
        self.init_status = InitStatus.DEFERRED_INITIALIZATION

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate sender and receiver

        Insure `sender <LearningProjection>` is a LearningMechanism or the OutputState of one.
        Insure `receiver <LearningProjection>` is a MappingProjection or the matrix ParameterState of one.
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

            if any(s in {OutputState, LearningSignal, LearningMechanism} for s in {sender, type(sender)}):
                # If it is the outputState of a LearningMechanism, check that it is a list or 1D np.array
                if isinstance(sender, OutputState):
                    if not isinstance(sender.value, (list, np.ndarray)):
                        raise LearningProjectionError("Sender for \'{}\' (OutputState of LearningMechanism \'{}\') "
                                                      "must be a list or 1D np.array".format(self.name, sender.name))
                    if not np.array(sender.value).ndim >= 1:
                        raise LearningProjectionError("OutputState of \'{}\' (LearningMechanism for \'{}\') must be "
                                                      "an ndarray with dim >= 1".format(sender.owner.name, self.name))
                # If specification is a LearningMechanism class, pass (it will be instantiated in _instantiate_sender)
                elif inspect.isclass(sender) and issubclass(sender,  LearningMechanism):
                    pass

            else:
                raise LearningProjectionError("The sender arg for {} ({}) must be a LearningMechanism, "
                                              "the OutputState or LearningSignal of one, or a reference to the class"
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
        if not iscompatible(self.value, self.receiver.value):
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
        if self.init_status is InitStatus.DEFERRED_INITIALIZATION:
            return self.init_status

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
        return self.sender.value

    @property
    def weight_change_matrix(self):
        return self.value

    @weight_change_matrix.setter
    def weight_change_matrix(self,assignment):
        self.value = assignment
