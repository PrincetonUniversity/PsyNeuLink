# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  ControlProjection *********************************************************

"""
.. _ControlProjection_Overview:

Overview
--------

A ControlProjection is a :doc:`projection <Projection> that projects to the :doc:`parameterState <ParameterState>`
of a mechanism.  It takes the value of an outputState of another mechanism (e.g., usually a :doc:`ControlMechanism`),
and uses it to modify the value of the parameter associated with the parameterState to which it projects.

.. _ControlProjection_Creation:

Creating a ControlProjection
----------------------------

A ControlProjection can be created using any of the standard ways to :ref:`create a projection <Projection_Creation>`,
or by including it in the :ref:`specification of a parameter <ParameterState_Specifying_Parameters>` for a mechanism,
MappingProjection, or the ``function`` of either of these.  If a ControlProjection is created using its constructor on
its own, the ``receiver`` argument must be specified.  If it is included in a parameter specification,
the parameterState for the parameter being specified will be assigned as the ControlProjection's ``receiver``.  If
its ``sender`` is not specified, its assignment depends on the ``receiver``.  If the receiver belongs to a mechanism
that is part of a system, then the ControlProjection's ``sender`` is assigned to an outputState of the system's
:ref:`controller  <System_Execution_Control>`. Otherwise, the ``sender`` is assigned to the outputState of a
:any:`DefaultControlMechanism`.

.. _ControlProjection_Structure:

Structure
---------

A ControlProjection has the same structure as a :doc:`Projection`.  Its
:py:data:`sender <Projection.Projection_Base.sender>` can be the outputState of any mechanism, but is generally a
:doc:`ControlMechansm`, and commonly the :doc:`ControlSignal` of an :doc:`EVCMechanism.`.  Its
:py:data:`receiver <Projection.Projection_Base.receiver>` is always the :doc:`paramterState <ParameterState>` of a
mechanism or :doc:`MappingProjection`, that is associated with a parameter of either the parameterState's owner or
it owner's ``function``.  The ``function`` of a ControlProjection is, by default, the identity function;  that is,
it uses the value of its sender to modify the value of the parameter that it controls.


.. _ControlProjection_Execution:

Execution
---------

A ControlProjection uses its ``function`` to assign its value from the one received from its :py:data:`sender`*[]:
This is used by the parmaterState to which it projects to modify the corresponding parameter of the
parameterState's owner or its owner's ``function``.

.. note::
   The changes in a parameter in response to the execution of a ControlProjection are not applied until the
   mechanism that receives the projection are next executed; see Lazy_Evaluation for an explanation of "lazy"
   updating).

.. _ControlProjection_Class_Reference:


Class Reference
---------------

"""

from PsyNeuLink.Components import DefaultController
# from Globals.Defaults import *
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Functions.Function import *

projection_keywords.update({CONTROL_PROJECTION})
parameter_keywords.update({CONTROL_PROJECTION})

class ControlProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class ControlProjection(Projection_Base):
    """
    ControlProjection(                                \
     sender=None,                                     \
     receiver=None,                                   \
     function=Linear                                  \
     params=None,                                     \
     name=None,                                       \
     prefs=None)

     Implements a projection that controls the parameter of a mechanism's function.

    COMMENT:
        Description:
            The ControlProjection class is a type in the Projection category of Component.
            It implements a projection to the parameterState of a mechanism that modifies a parameter of its function.
            It:
               - takes an allocation (scalar) as its input (self.variable)
               - uses self.function (params[FUNCTION]) to compute intensity based on allocation from self.sender,
                   used by self.receiver.owner to modify a parameter of self.receiver.owner.function.

        ** MOVE:
        ProjectionRegistry:
            All ControlProjections are registered in ProjectionRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

        Class attributes:
            + color (value): for use in interface design
            + classPreference (PreferenceSet): ControlProjectionPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + paramClassDefaults:
                FUNCTION:Linear,
                FUNCTION_PARAMS:{SLOPE: 1, INTERCEPT: 0},  # Note: this implements identity function
                PROJECTION_SENDER: DefaultController, # ControlProjection (assigned to class ref in __init__ module)
                PROJECTION_SENDER_VALUE: [defaultControlAllocation],
                CONTROL_SIGNAL_COST_OPTIONS:ControlSignalCostOptions.DEFAULTS,
                kwControlSignalLogProfile: ControlSignalLog.DEFAULTS,
                ALLOCATION_SAMPLES: DEFAULT_ALLOCATION_SAMPLES,
            + paramNames = paramClassDefaults.keys()
            + costFunctionNames = paramClassDefaults[kwControlSignalCostFunctions].keys()
    COMMENT

    Arguments
    ---------

    sender : Optional[Mechanism or OutputState]
        the source of the allocation for the ControlProjection;  usually an outputState of a :doc:`ControlMechanism`.
        If it is not specified, the :doc:`DefaultControlMechanism` for the system to which the receiver belongs
        will be assigned.

    receiver : Optional[Mechanism or ParameterState]
        the parameterState associated with the parameter of a function to be controlled.  This must be specified,
        or be able to be determined by the context in which the ControlProjection is created or assigned.

    function : TransferFunction : default Linear
        converts the value of the ControlProjection's :pyd:data:`sender`  into its ``value``.

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the projection, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Component` for specification of a params dict).
        By default, it contains an entry for the projection's default ``function`` and cost function assignments.

    name : str : default ControlProjection-<index>
        a string used for the name of the ControlProjection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Projection.classPreferences]
        the PreferenceSet for the ControlProjection.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

    sender : OutputState of ControlProjection
        mechanism that provides the current input for the ControlProjection (usuall a :doc:`ControlMechanisms`).

    receiver : ParameterState of Mechanism
        :doc:`parameterState <ParameterState>` for the parameter to be modified by ControlProjection.

    value : float
        during initialization, assigned keyword string (either INITIALIZING or DEFERRED_INITIALIZATION);
        during execution, returns the ``value`` of the ControlProjection.

    name : str : default ControlProjection-<index>
        the name of the ControlProjection.
        Specified in the name argument of the call to create the projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the PreferenceSet for projection.
        Specified in the prefs argument of the call to create the projection;
        if it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).


    """

    color = 0

    componentType = CONTROL_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    variableClassDefault = 0.0

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_SENDER: DefaultController,
        PROJECTION_SENDER_VALUE: defaultControlAllocation})

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 function=Linear,
                 control_signal:tc.optional(dict)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function)

        # If receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        if not receiver:
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = self
            self.init_args['name'] = name
            # Delete thi as it has to be moved to params dict (and so will not be recognized by Projection.__init__)
            del self.init_args[CONTROL_SIGNAL]

            # Flag for deferred initialization
            self.value = DEFERRED_INITIALIZATION
            return

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        # super(ControlSignal_Base, self).__init__(sender=sender,
        super(ControlProjection, self).__init__(sender=sender,
                                            receiver=receiver,
                                            params=params,
                                            name=name,
                                            prefs=prefs,
                                            context=self)


    def _instantiate_sender(self, params=None, context=None):
# FIX: NEEDS TO BE BETTER INTEGRATED WITH super()._instantiate_sender
        """Check if DefaultController is being assigned and if so configures it for the requested ControlProjection

        If self.sender is a Mechanism, re-assign to <Mechanism>.outputState
        Insure that sender.value = self.variable

        This method overrides the corresponding method of Projection, before calling it, to check if the
            DefaultController is being assigned as sender and, if so:
            - creates projection-dedicated inputState and outputState in DefaultController
            - puts them in DefaultController's inputStates and outputStates attributes
            - lengthens variable of DefaultController to accommodate the ControlProjection channel
            - updates value of DefaultController (in resposne to new variable)
        Notes:
            * the default execute method of DefaultController simply maps the inputState value to the outputState
            * the params arg is assumed to be a dictionary of params for the controlSignal of the ControlMechanism

        :return:
        """

        if isinstance(self.sender, Process):
            raise ProjectionError("Illegal attempt to add a ControlProjection from a Process {0} "
                                  "to a mechanism {0} in pathway list".format(self.name, self.sender.name))

        # If sender is a class:
        # - assume it is Mechanism or State class ref (as validated in _validate_params)
        # - implement default sender of the corresponding type
        if inspect.isclass(self.sender):
            # self.sender = self.paramsCurrent[PROJECTION_SENDER](self.paramsCurrent[PROJECTION_SENDER_VALUE])
# FIX 6/28/16:  IF CLASS IS ControlMechanism SHOULD ONLY IMPLEMENT ONCE;  THEREAFTER, SHOULD USE EXISTING ONE
            self.sender = self.sender(self.paramsCurrent[PROJECTION_SENDER_VALUE])

# FIX:  THE FOLLOWING CAN BE CONDENSED:
# FIX:      ONLY TEST FOR ControlMechanism_Base (TO IMPLEMENT PROJECTION)
# FIX:      INSTANTATION OF OutputState WILL BE HANDLED IN CALL TO super._instantiate_sender
# FIX:      (CHECK TO BE SURE THAT THIS DOES NOT MUCK UP _instantiate_control_projection FOR ControlMechanism)
        # If sender is a Mechanism (rather than a State) object, get (or instantiate) its State
        #    (Note:  this includes ControlMechanism)
        if isinstance(self.sender, Mechanism):
            # If sender is a ControlMechanism, call it to instantiate its controlSignal projection
            from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
            from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlSignal import ControlSignalError
            if isinstance(self.sender, ControlMechanism_Base):
                # MODIFIED 12/23/16 NEW:
                #   [TRY AND EXCEPT IS NEW, AS IS ADDITION OF param ARG IN CALL TO _instantiate_control_projection]
                try:
                    self.sender._instantiate_control_projection(self, params=params, context=context)
                except ControlSignalError as error_msg:
                    raise FunctionError("Error in attempt to specify controlSignal for {} of {}".
                                        format(self.name, self.receiver.owner.name, error_msg))
                # MODIFIED 12/23/16 END

        # Call super to instantiate sender

        super(ControlProjection, self)._instantiate_sender(context=context)


    def _instantiate_receiver(self, context=None):
        # FIX: THIS NEEDS TO BE PUT BEFORE _instantate_function SINCE THAT USES self.receiver
        """Handle situation in which self.receiver was specified as a Mechanism (rather than State)

        Overrides Projection._instantiate_receiver, to require that if the receiver is specified as a Mechanism, then:
            the receiver Mechanism must have one and only one ParameterState;
            otherwise, passes control to Projection._instantiate_receiver for validation

        :return:
        """
        if isinstance(self.receiver, Mechanism):
            # If there is just one param of ParameterState type in the receiver Mechanism
            # then assign it as actual receiver (which must be a State);  otherwise, raise exception
            from PsyNeuLink.Components.States.ParameterState import ParameterState
            if len(dict((param_name, state) for param_name, state in self.receiver.paramsCurrent.items()
                    if isinstance(state, ParameterState))) == 1:
                receiver_parameter_state = [state for state in dict.values()][0]
                # Reassign self.receiver to Mechanism's parameterState
                self.receiver = receiver_parameter_state
                # # Add self as projection to that parameterState
                # # IMPLEMENTATION NOTE:
                # #   THIS SHOULD REALLY BE HANDLED BY THE Mechanism.add_projection METHOD, AS IT IS FOR inputStates
                # # # MODIFIED 6/22/16 OLD:
                # # self.receiver.receivesFromProjections.append(self)
                # # MODIFIED 6/22/16 NEW:
                # self.receiver.add_projection(projection=self, state=receiver_parameter_state, context=context)
            else:
                raise ControlProjectionError("Unable to assign ControlProjection ({0}) from {1} to {2}, "
                                         "as it has several parameterStates;  must specify one (or each) of them"
                                         " as receiver(s)".
                                         format(self.name, self.sender.owner, self.receiver.name))
        # else:
        super(ControlProjection, self)._instantiate_receiver(context=context)

    def execute(self, params=None, clock=CentralClock, time_scale=None, context=None):
    # def execute(self, params=None, clock=CentralClock, time_scale=TimeScale.TRIAL, context=None):

        self.variable = self.sender.value
        self.value = self.function(variable=self.variable, params=params, time_scale=time_scale, context=context)
        return self.value
