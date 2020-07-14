# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  LearningProjection **********************************************************

"""

Contents
--------

  * `LearningProjection_Overview`
  * `LearningProjection_Creation`
      - `LearningProjection_Deferred_Initialization`
      - `LearningProjection_Sender`
      - `LearningProjection_Function_and_Learning_Rate`
      - `LearningProjection_Receiver`
  * `LearningProjection_Structure`
  * `LearningProjection_Execution`
  * `LearningProjection_Class_Reference`


.. _LearningProjection_Overview:

Overview
--------

A LearningProjection is a type of `ModulatoryProjection <ModulatoryProjection>` that projects from a `LearningMechanism`
to the *MATRIX* `ParameterPort` of a `MappingProjection`.  It takes the `value <LearningSignal.value>` of a
`LearningSignal` of a `LearningMechanism`, and uses it to modify the value of the `matrix <MappingProjection.matrix>`
parameter of that MappingProjection.  All of the LearningProjections in a System, along with its other `learning
components <LearningMechanism_Learning_Configurations>`, can be displayed using the System's `show_graph
<System.show_graph>` method with its **show_learning** argument assigned as `True`.

.. _LearningProjection_Creation:

Creating a LearningProjection
------------------------------------

A LearningProjection can be created using any of the standard ways to `create a Projection <Projection_Creation>`,
or by including it in a `tuple <MappingProjection_Tuple_Specification>` that specifies the `matrix
<MappingProjection.matrix>` parameter of a `MappingProjection`.  LearningProjections are also created automatically,
along with the other `Components required for learning <LearningMechanism_Learning_Configurations>`, when learning is
specified for a `Process <Process_Learning_Sequence>` or a `System <System_Execution_Learning>`.

If a LearningProjection is created explicitly (using its constructor), and its **receiver** argument is not specified,
its initialization is `deferred <LearningProjection_Deferred_Initialization>`.  If it is included in the `matrix
specification <MappingProjection_Tuple_Specification>` for a `MappingProjection`, the *MATRIX* `ParameterPort` for
the MappingProjection will be assigned as the LearningProjection's `receiver <LearningProjection.receiver>`.  If its
**sender** argument is not specified, its assignment depends on the **receiver**.  If the **receiver** belongs to a
MappingProjection that projects between two Mechanisms that are both in the same `Process <Process_Learning_Sequence>`
or `System <System_Execution_Learning>`, then the LearningProjection's `sender <LearningProjection.sender>` is assigned
to a `LearningSignal` of the `LearningMechanism` for the MappingProjection. If there is none, it is `created
<LearningMechanism_Creation>` along with any other components needed to implement learning for the MappingProjection
(see `LearningMechanism_Learning_Configurations`). Otherwise, the LearningProjection's initialization is `deferred
<LearningProjection_Deferred_Initialization>`.

.. _LearningProjection_Deferred_Initialization:

*Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~

When a LearningProjection is created, its full initialization is `deferred <Component_Deferred_Init>` until its
`sender <LearningProjection.sender>` and `receiver <LearningProjection.receiver>` have been fully specified.  This
allows a LearningProjection to be created before its `sender <LearningProjection.sender>` and/or `receiver
<LearningProjection.receiver>` have been created (e.g., before them in a script), by calling its constructor without
specifying its **sender** or **receiver** arguments. However, for the LearningProjection to be operational,
initialization must be completed by calling its `deferred_init` method.  This is not necessary if the
LearningProjection is included in a `tuple specification <MappingProjection_Tuple_Specification>` for the `matrix
<MappingProjection.matrix>` parameter of a `MappingProjection`, in which case deferred initialization is completed
automatically when the `LearningMechanism` associated with that MappingProjection is created for the `Process` or
`System` to which it belongs (see `LearningMechanism_Creation`).

.. _LearningProjection_Structure:

Structure
---------

.. _LearningProjection_Sender:

*Sender*
~~~~~~~~

The `sender <LearningProjection.sender>` of a LearningProjection is a `LearningSignal` of a `LearningMechanism`,
The `value <LearningSignal.value>` of the `sender <LearningProjection.sender>` -- a matrix of weight changes --
is used by the LearningProjection as its `variable <LearningProjection.variable>`;  this is also assigned to its
`learning_signal <LearningProjection.learning_signal>` attribute, and serves as the input to the LearningProjection's
`function <LearningProjection.function>`.

.. _LearningProjection_Function_and_Learning_Rate:

*Function and learning_rate*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default `function <LearningProjection.function>` of a LearningProjection is an identity function (`Linear` with
**slope**\\ =1 and **intercept**\\ =0).  However, its result can be modified by the LearningProjection's `learning_rate
<LearningProjection.learning_rate>` parameter (specified in the **learning_rate** argument of its constructor).
If specified, it is applied multiplicatively to the output of the LearningProjection's `function
<LearningProjection.function>`, and the result is assigned to the LearningProjection's `value
<LearningProjection.value>` and `weight_change_matrix <LearningProjection.weight_change_matrix>` attributes.  Thus,
the LearningProjection's `learning_rate <LearningProjection.learning_rate>` parameter can be used to modulate the
`learning_signal <LearningProjection.learning_signal>` it receives, in addition to (and on top of) the effects of the
`learning_rate <LearningMechanism.learning_rate>` for the `LearningMechanism` from which it receives the
`learning_signal <LearningProjection.learning_signal>`. Specification of the `learning_rate
<LearningProjection.learning_rate>` for a LearningProjection supersedes any specification(s) of the
:keyword:`learning_rate` for any `Process <Process_Learning_Sequence>` and/or `System <System_Learning>` to which the
Projection belongs (see `learning_rate <LearningMechanism_Learning_Rate>` for additional details).  However, its
`learning_rate <LearningProjection.learning_rate>` can be specified by the `LearningSignal
<LearningSignal_Learning_Rate>` that is its `sender <LearningProjection.sender>`;  that specification takes precedence
over the direct specification of the `learning_rate <LearningProjection.learning_rate>` for the LearningProjection
(i.e., in the **learning_rate** argument of its constructor, or by direct assignment of a value to the attribute).  If a
`learning_rate <LearningProjection.learning_rate>` is not specified for the LearningProjection, then the result of its
`function <LearningProjection.function>` is assigned unmodified as the LearningProjection's `value
<LearningProjection.value>` (and `weight_change_matrix <LearningProjection.weight_change_matrix>` attributes.

.. _Learning_Weight_Exponent:

*Weight and Exponent*
~~~~~~~~~~~~~~~~~~~~~

Every LearningProjection has a `weight <Projection_Base.weight>` and `exponent <Projection_Base.exponent>`
attribute that are applied to its `value <LearningProjection.value>` before it is combined  with other
LearningProjections that modify the `ParameterPort` for the `matrix <MappingProjection.matrix>` parameter of the
`MappingProjection` to which they project (see description under `Projection <Projection_Weight_Exponent>` for
additional details).

.. note::
   The `weight <Projection_Base.weight>` and `exponent <Projection_Base.exponent>` attributes of a LearningProjection
   are not commonly used, and are implemented largely for generality and compatibility with other types of `Projection`.
   They are distinct from, and are applied in addition to the LearningProjection's `learning_rate
   <LearningProjection.learning_rate>` attribute.  As noted under  `Projection <Projection_Weight_Exponent>`, they are
   not normalized and thus their effects aggregate if a ParameterPort receives one or more LearningProjections with
   non-default values of their  `weight <Projection_Base.weight>` and `exponent <Projection_Base.exponent>` attributes.

.. _LearningProjection_Receiver:

*Receiver*
~~~~~~~~~~

The `receiver <LearningProjection.receiver>` of a LearningProject is the *MATRIX* `ParameterPort` of a
`MappingProjection`, that uses the `weight_change_matrix <LearningProjection.weight_change_matrix>` provided by the
LearningProjection to modify the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` being
learned.

.. _LearningProjection_Execution:

Execution
---------

A LearningProjection cannot be executed directly.  It's execution is determined by its `learning_enabled
<LearningProjection.learning_enabled>` attribute.  If that is False`, it is never executed.  If it is True or
*ONLINE*, is executed when the *MATRIX* ParameterPort to which it projects is updated.  This occurs when the
`learned_projection <LearningProjection.learned_projection>` (the `MappingProjection` to which the *MATRIX*
ParameterPort belongs) is updated. Note that these events occur only when the ProcessingMechanism that receives the
`learned_projection <LearningProjection.learned_projection>` is executed (see `Lazy Evaluation
<Component_Lazy_Updating>` for an explanation of "lazy" updating).  If `learning_enabled
<LearningProjection.learning_enabled>` is *AFTER*, then LearningProjection is executed at the end of the `TRIAL
<TimeScale.TRIAL>` of the Composition to which it belongs.  When the LearningProjection is executed, its `function
<LearningProjection.function>` gets the `learning_signal <LearningProjection.learning_signal>` from its `sender
<LearningProjection.sender>` and conveys that to its `receiver <LearningProjection.receiver>`, possibly modified by
a `learning_rate <LearningProjection.learning_rate>` if that is specified for it or its `sender
<LearningProjection.sender>` (see `above <LearningProjection_Function_and_Learning_Rate>`).

.. note::
   The changes to the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection` in response to the
   execution of a LearningProjection are not applied until the `Mechanism <Mechanism>` that receives MappingProjection
   are next executed; see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).


.. _LearningProjection_Class_Reference:

Class Reference
---------------

"""

import inspect
import warnings

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.learningfunctions import BackPropagation
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import ERROR_SIGNAL, LearningMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import Projection_Base, projection_keywords
from psyneulink.core.components.shellclasses import ShellClass
from psyneulink.core.components.ports.modulatorysignals.learningsignal import LearningSignal
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.keywords import \
    CONTEXT, FUNCTION, FUNCTION_PARAMS, INTERCEPT, LEARNING, LEARNING_PROJECTION, LEARNING_SIGNAL, \
    MATRIX, PARAMETER_PORT, PARAMETER_PORTS, PROJECTION_SENDER, SLOPE, ONLINE, AFTER
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import iscompatible, parameter_spec

__all__ = [
    'DefaultTrainingMechanism', 'LearningProjection', 'LearningProjectionError', 'WT_MATRIX_RECEIVERS_DIM', 'WT_MATRIX_SENDER_DIM',
]
# Parameters:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

WT_MATRIX_SENDER_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

DefaultTrainingMechanism = ObjectiveMechanism


class LearningProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _learning_signal_getter(owning_component=None, context=None):
    return owning_component.sender.parameters.value._get(context)


def _learning_signal_setter(value, owning_component=None, context=None):
    owning_component.sender.parameters.value._set(value, context)
    return value


class LearningProjection(ModulatoryProjection_Base):
    """
    LearningProjection(                \
                 sender=None,          \
                 receiver=None,        \
                 error_function,       \
                 learning_function,    \
                 learning_rate=None,   \
                 learning_enabled=None \
                 weight=None,          \
                 exponent=None,        \
                 params=None,          \
                 name=None,            \
                 prefs=None)

    Subclass of `ModulatoryProjection <ModulatoryProjection>` that modulates the value of a `ParameterPort` for the
    `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`.
    See `Projection <ModulatoryProjection_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    sender : Optional[LearningMechanism or LearningSignal]
        specifies the source of the `learning_signal <LearningProjection.learning_signal>` for the LearningProjection;
        if it is not specified, and cannot be `inferred from context <LearningProjection_Creation>`, initialization is
        `deferred <LearningProjection_Deferred_Initialization>`.

    receiver : Optional[MappingProjection or ParameterPort for matrix parameter of one]
        specifies the *MATRIX* `ParameterPort` (or the `MappingProjection` that owns it) for the `matrix
        <MappingProjection.matrix>` of the `learned_projection <LearningProjection.learned_projection>` to be
        modified by the LearningProjection; if it is not specified, and cannot be `inferred from context
        <LearningProjection_Creation>`, initialization is `deferred <LearningProjection_Deferred_Initialization>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies the function used to convert the `learning_signal` to the `weight_change_matrix
        <LearningProjection.weight_change_matrix>`, prior to applying the `learning_rate
        <LearningProjection.learning_rate>`.

    error_function : Optional[Function or function] : default LinearCombination(weights=[[-1], [1]])
        specifies a function to be used by the `TARGET Mechanism <LearningMechanism_Targets>` to compute the error
        used for learning.  Since the `TARGET` Mechanism is a `ComparatorMechanism`, its function must have a `variable
        <Function.variable>` with two items, that receives its values from the *SAMPLE* and *TARGET* InputPorts of the
        ComparatorMechanism.

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
        :keyword:`learning_rate` for any `Process <Process_Learning_Sequence>` and/or `System <System_Learning>` to
        which the LearningProjection belongs, and is applied in addition to any effects of the `learning_rate
        <LearningMechanism.learning_rate>` for the `LearningMechanism` from which the LearningProjection receives its
        `learning_signal <LearningProjection.learning_signal>` (see `LearningProjection_Function_and_Learning_Rate` for
        additional details).

    learning_enabled : Optional[bool or Enum[ONLINE|AFTER]] : default : None
        determines whether or when the `value <LearningProjection.value>` of the LearningProjection is used to modify
        the `learned_projection <LearningProjection.learned_projection>` when the latter is executed (see
        `learning_enabled <LearningProjection.learning_enabled>` for additional details).


    Attributes
    ----------

    sender : LearningSignal
        source of `learning_signal <LearningProjection.learning_signal>`
        (see `LearningProjection_Sender` for additional details).

    receiver : MATRIX ParameterPort of a MappingProjection
        *MATRIX* `ParameterPort` for the `matrix <MappingProjection.MappingProjection.matrix>` parameter of the
        `learned_projection <LearningProjection.learned_projection>` (see `LearningProjection_Receiver` for additional
        details).

    learned_projection : MappingProjection
        the `MappingProjection` to which LearningProjection's `receiver <LearningProjection.receiver>` belongs.

    variable : 2d np.array
        same as `learning_signal <LearningProjection.learning_signal>`.

    learning_enabled : bool or Enum[ONLINE|AFTER]
        determines whether or when the `value <LearningProjection.value>` of the LearningProjection is used to modify
        the `learned_projection <LearningProjection.learned_projection>` when the latter is executed;  by default, its
        value is set to the value of the `learning_enabled <LearningMechanism.learning_enabled>` attribute of the
        `LearningMechanism` to which the LearningProjection's `sender <LearningProjection.sender>` belongs; however
        it can be overridden using the **learning_enabled** argument of the LearningProjection's constructor and/or
        by assignment after it is constructed (see `learning_enabled <LearningMechanism.learning_enabled>` for
        additional details).

    learning_signal : 2d np.array
        the `value <LearningSignal.value>` of the LearningProjection's `sender <LearningProjection.sender>`: a matrix of
        weight changes calculated by the `LearningMechanism` to which the `sender <LearningProjection.sender>` belongs;
        rows correspond to the `sender <MappingProjection.sender>` of the `learned_projection <LearningProjection>`,
        and columns to its `receiver <MappingProjection.receiver>` (i.e., the input and output of the
        `learned_projection <LearningProjection>`, respectively).

    function : Function
        assigns the `learning_signal` received from the `sender <LearningProjection.sender>` to the
        LearningProjection's `value <LearningProjection.value>`, possibly modified by its `learning_rate
        <LearningProjection.learning_rate>`; the default in an identity function.

    learning_rate : float or `None`
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

    """

    componentType = LEARNING_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    class sockets:
        sender=[LEARNING_SIGNAL]
        receiver=[PARAMETER_PORT]

    class Parameters(ModulatoryProjection_Base.Parameters):
        """
            Attributes
            ----------

                value
                    see `value <LearningProjection.value>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True

                error_function
                    see `error_function <LearningProjection.error_function>`

                    :default value: `LinearCombination`(weights=numpy.array([[-1], [ 1]]))
                    :type: `Function`

                function
                    see `function <LearningProjection.function>`

                    :default value: `Linear`
                    :type: `Function`

                learning_enabled
                    see `learning_enabled <LearningProjection.learning_enabled>`

                    :default value: None
                    :type:

                learning_function
                    see `learning_function <LearningProjection.learning_function>`

                    :default value: `BackPropagation`
                    :type: `Function`

                learning_rate
                    see `learning_rate <LearningProjection.learning_rate>`

                    :default value: None
                    :type:

                learning_signal
                    see `learning_signal <LearningProjection.learning_signal>`

                    :default value: None
                    :type:
                    :read only: True
        """
        value = Parameter(np.array([0]), read_only=True, aliases=['weight_change_matrix'], pnl_internal=True)
        function = Parameter(Linear, stateful=False, loggable=False)
        error_function = Parameter(LinearCombination(weights=[[-1], [1]]), stateful=False, loggable=False, reference=True)
        learning_function = Parameter(BackPropagation, stateful=False, loggable=False, reference=True)
        learning_rate = Parameter(None, modulable=True)
        learning_signal = Parameter(None, read_only=True,
                                    getter=_learning_signal_getter,
                                    setter=_learning_signal_setter,
                                    pnl_internal=True)
        learning_enabled = None


    projection_sender = LearningMechanism

    @tc.typecheck
    def __init__(self,
                 sender:tc.optional(tc.any(LearningSignal, LearningMechanism))=None,
                 receiver:tc.optional(tc.any(ParameterPort, MappingProjection))=None,
                 error_function:tc.optional(is_function_type)=None,
                 learning_function:tc.optional(is_function_type)=None,
                 # FIX: 10/3/17 - TEST IF THIS OK AND REINSTATE IF SO
                 # learning_signal_params:tc.optional(dict)=None,
                 learning_rate:tc.optional(tc.any(parameter_spec))=None,
                 learning_enabled:tc.optional(tc.any(bool, tc.enum(ONLINE, AFTER)))=None,
                 weight=None,
                 exponent=None,
                 params:tc.optional(dict)=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs
                 ):

        # IMPLEMENTATION NOTE:
        #     the error_function and learning_function arguments are implemented to preserve the ability to pass
        #     error function and learning function specifications from the specification of a LearningProjection (used
        #     to implement learning for a MappingProjection, e.g., in a tuple) to the LearningMechanism responsible
        #     for implementing the function; and for specifying the default LearningProjection for a Process.
        # If receiver has not been assigned, defer init to Port.instantiate_projection_to_state()
        if sender is None or receiver is None:
            # Flag for deferred initialization
            self.initialization_status = ContextFlags.DEFERRED_INIT

            # parameters should be passed through methods like
            # instantiate_sender instead of grabbed from attributes like this
            self._learning_function = learning_function
            self._learning_rate = learning_rate
            self._error_function = error_function

        # replaces similar code in _instantiate_sender
        try:
            if sender.owner.learning_rate is not None:
                learning_rate = sender.owner.learning_rate
        except AttributeError:
            pass

        super().__init__(sender=sender,
                         receiver=receiver,
                         weight=weight,
                         exponent=exponent,
                         params=params,
                         name=name,
                         prefs=prefs,
                         error_function=error_function,
                         learning_function=learning_function,
                         learning_rate=learning_rate,
                         learning_enabled=learning_enabled,
                         **kwargs)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate sender and receiver

        Insure `sender <LearningProjection>` is a LearningMechanism or the OutputPort of one.
        Insure `receiver <LearningProjection>` is a MappingProjection or the matrix ParameterPort of one.
        """

        # IMPLEMENTATION NOTE: IS TYPE CHECKING HERE REDUNDANT WITH typecheck IN __init__??

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if self.initialization_status == ContextFlags.INITIALIZING:
            # VALIDATE SENDER
            sender = self.sender
            if isinstance(sender, LearningMechanism):
                if len(sender.learning_signals) > 1:
                    raise LearningProjectionError("PROGRAM ERROR: {} has more than one LearningSignal "
                                                  "which is not currently supported".format(sender.name))
                sender = self.sender = sender.learning_signals[0]

            if any(s in {OutputPort, LearningSignal, LearningMechanism} for s in {sender, type(sender)}):
                # If it is the outputPort of a LearningMechanism, check that it is a list or 1D np.array
                if isinstance(sender, OutputPort):
                    if not isinstance(sender.value, (list, np.ndarray)):
                        raise LearningProjectionError("Sender for \'{}\' (OutputPort of LearningMechanism \'{}\') "
                                                      "must be a list or 1D np.array".format(self.name, sender.name))
                    if not np.array(sender.value).ndim >= 1:
                        raise LearningProjectionError("OutputPort of \'{}\' (LearningMechanism for \'{}\') must be "
                                                      "an ndarray with dim >= 1".format(sender.owner.name, self.name))
                # If specification is a LearningMechanism class, pass (it will be instantiated in _instantiate_sender)
                elif inspect.isclass(sender) and issubclass(sender,  LearningMechanism):
                    pass

            else:
                raise LearningProjectionError("The sender arg for {} ({}) must be a LearningMechanism, "
                                              "the OutputPort or LearningSignal of one, or a reference to the class"
                                              .format(self.name, sender.name))


            # VALIDATE RECEIVER
            receiver = self.receiver
            if isinstance(receiver, MappingProjection):
                try:
                    receiver = self.receiver = receiver._parameter_ports[MATRIX]
                except KeyError:
                    raise LearningProjectionError("The MappingProjection {} specified as the receiver for {} "
                                                  "has no MATRIX ParameterPort".format(receiver.name, self.name))
            if not any(s in {ParameterPort, MappingProjection} for s in {receiver, type(receiver)}):
                raise LearningProjectionError("The receiver arg for {} must be a MappingProjection "
                                              "or the MATRIX parameterPort of one."
                                              .format(PROJECTION_SENDER, sender, self.name, ))

    def _instantiate_sender(self, sender, context=None):
        """Instantiate LearningMechanism
        """

        # LearningMechanism was specified by class or was not specified,
        #    so call composition for "automatic" instantiation of a LearningMechanism
        # Note: this also instantiates an ObjectiveMechanism if necessary and assigns it the necessary projections

        # assignment to attribute necessary because of uses in _instantiate_learning_components
        self.sender = sender

        if not isinstance(self.sender, (OutputPort, LearningMechanism)):
            warnings.warn("Instantiation of a LearningProjection outside of a Composition is tricky!")

        if isinstance(self.sender, OutputPort) and not isinstance(self.sender.owner, LearningMechanism):
            raise LearningProjectionError("Sender specified for LearningProjection {} ({}) is not a LearningMechanism".
                                          format(self.name, self.sender.owner.name))

        # This assigns LearningProjection as an outgoing projection from the LearningMechanism's LearningSignal
        #    OutputPort and formats the LearningProjection's defaults.variable to be compatible with
        #    the LearningSignal's value
        super()._instantiate_sender(self.sender, context=context)

    def _instantiate_receiver(self, context=None):
        """Validate that receiver has been assigned and is compatible with the output of function

        Set learning_enabled to value of receiver if it was not otherwise specified in the constructor

        Notes:
        * _validate_params verifies that receiver is a parameterPort for the matrix parameter of a MappingProjection.
        * _super()._instantiate_receiver verifies that the projection has not already been assigned to the receiver.

        """

        super()._instantiate_receiver(context=context)

        # Insure that the learning_signal is compatible with the receiver's weight matrix
        if not iscompatible(self.defaults.value, self.receiver.defaults.variable):
            raise LearningProjectionError("The learning_signal of {} ({}) is not compatible with the matrix of "
                                          "the MappingProjection ({}) to which it is being assigned ({})".
                                          format(self.name,
                                                 self.defaults.value,
                                                 self.receiver.defaults.value,
                                                 self.receiver.owner.name))

        # Insure that learning_signal has the same shape as the receiver's weight matrix
        try:
            receiver_weight_matrix_shape = np.array(self.receiver.defaults.value).shape
        except TypeError:
            receiver_weight_matrix_shape = 1
        try:
            learning_signal_shape = np.array(self.defaults.value).shape
        except TypeError:
            learning_signal_shape = 1


        # FIX: SHOULD TEST WHETHER IT CAN BE USED, NOT WHETHER IT IS THE SAME SHAPE
        learning_mechanism = self.sender.owner
        learned_projection = self.receiver.owner

        # Set learning_enabled to value of its LearningMechanism sender if it was not specified in the constructor
        if self.learning_enabled is None:
            self.learning_enabled = self.parameters.learning_enabled.default_value = learning_mechanism.learning_enabled

        learned_projection.learning_mechanism = learning_mechanism
        learned_projection.has_learning_projection = self

    def _execute(self, variable, context=None, runtime_params=None):
        """
        :return: (2D np.array) self.weight_change_matrix
        """
        runtime_params = runtime_params or {}

        # Pass during initialization (since has not yet been fully initialized
        if self.initialization_status == ContextFlags.DEFERRED_INIT:
            return self.initialization_status

        if variable is not None:
            learning_signal = variable
        else:
            learning_signal = self.sender.parameters.value._get(context)
        matrix = self.receiver.parameters.value._get(context)
        # If learning_signal is lower dimensional than matrix being trained
        #    and the latter is a diagonal matrix (square, with values only along the main diagonal)
        #    and the learning_signal is the same as the matrix,
        #    then transform the learning_signal into a diagonal matrix of the same dimension as the matrix
        # Otherwise, if the learning_signal and matrix are not the same shape,
        #    try expanding dim of learning_signal by making each of its items (along axis 0) an array
        # Example:
        #    If learning_signal is from a LearningMechanism that uses Reinforcement Function, then it is a 1d array:
        #        if the matrix being modified is a 2d array, then convert the learning_signal to a 2d diagonal matrix;
        #        if the matrix being modified is a 1d array, then expand it so that each item is a 1d array
        # NOTE: The current version is only guaranteed to work learning_signal.ndim =1 and matrix.ndim = 2
        if (
                (learning_signal.ndim < matrix.ndim) and
                np.allclose(matrix,np.diag(np.diag(matrix))) and
                len(learning_signal)==len(np.diag(matrix))):
            learning_signal = np.diag(learning_signal)
        elif learning_signal.shape != matrix.shape:
            # Convert 1d array into 2d array to match format of a Projection.matrix
            learning_signal = np.expand_dims(learning_signal, axis=1)
            if learning_signal.shape != matrix.shape:
                raise LearningProjectionError("Problem modifying learning_signal from {} ({}) "
                                              "to match the matrix of {} it is attempting to modify ({})".
                                              format(self.sender.owner.name, learning_signal,
                                                     self.receiver.owner.name, matrix))

        # IMPLEMENTATION NOTE:  skip Projection._execute, as that uses self.sender.value as variable,
        #                       which undermines formatting of it (as learning_signal) above
        value = super(ShellClass, self)._execute(
            variable=learning_signal,
            context=context,
            runtime_params=runtime_params,

        )

        if self.initialization_status != ContextFlags.INITIALIZING and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, np.diag(value)))

        return value
