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

A ControlProjection implements a control signal used to modify the value of a parameter of a mechanism's function.  It
takes a value (an *allocation*) from a ControlMechanism (its ``sender``), and uses this to compute the control signal's
:py:data:`intensity <ControlProjection.intensity>`, which is assigned as the ControlProjection's value.  The
parameterState that receives the ControlProjection uses its value to regulate the value of a parameter of a
mechanism's ``function``. A ControlProjection also calculates a :py:data:`cost <ControlProjection.cost>` for the
control signal, based on its intensity  and/or its time course, that can be used by a ControlMechanism to adapt the
ControlProjection's :py:data:`allocation <ControlProjection.allocation>`.

.. _ControlProjection_Creation:

Creating a ControlProjection
----------------------------

A ControlProjection can be created using any of the standard ways to  :ref:`create a projection <Projection_Creation>`,
or by including it in the :ref:`specification of a parameter <ParameterState_Specifying_Parameters>` for a mechanism,
MappingProjection, or their ``function``.  If a ConrolProjection is created using its constructor on its own,
the ``receiver`` argument must be specified.  If it is included in a parameter specification, the parameterState for
the parameter being specified will be assigned as the ControlProjection's ``receiver``.  If its ``sender`` is not
specified, its assignment depends on the ``receiver``.  If the receiver belongs to a mechanism that is part of a
system, then the ControlProjection's ``sender`` is assigned to an outputState of the system's
:ref:`controller  <System_Execution_Control>`. Otherwise, the ``sender`` is assigned to the outputState of a
:any:`DefaultControlMechanism`.

The four functions used to calculate the :ref:`cost of a control signal <ControlProjection_Cost_Functions>`
can be specified  either in arguments to the ControlProjection's constructor, or in a ``params dictionary`` (see
:ref:`ControlProjection_Cost_Functions` below). A custom function can be assigned to any cost function, so long as
it accepts the appropriate type of value and returns a scalar.  Each of the cost functions can be
:ref:`enabled or disabled <ControlProjection_Toggle_Costs>`, to select which make contributions to the
ControlProjection's :py:data:`cost <ControlProjection.cost>`.  A cost function can also be permanently disabled for
its ControlProjection by assigning :keyword:`None` to the argument for that function in its constructor (or the
appropriate entry in its ``params`` dictionary). Cost functions that are permanently disabled in this way cannot be
re-enabled.

A ControlProjection takes an ``allocation_samples`` specification as its input.  This must be an array that
specifies the values of its :py:data:`allocation <ControlProjection.allocation>` that will be sampled by
ControlMechanisms that adaptively adjust ControlProjection allocations (e.g., :doc:`EVCMechanism`).  The default is
an array of values from 0.1 to 1.0 in steps of 0.1.

.. _ControlProjection_Structure:

Structure
---------

*Intensity*. The ControlProjection's ``function`` uses its :py:data:`allocation <ControlProjection.allocation>` to
calculate a control signal :py:data:`intensity <ControlProjection.intensity>`. The default is an identity function
``(Linear(slope=1, intercept=0))``: the ControlProjection sets its control signal
:py:data:`intensity <ControlProjection.intensity>` equal to its :py:data:`allocation <ControlProjection.allocation>`.
The ``function`` can be assigned another :py:doc:`TransferFunction`, or any other function that takes and returns a
scalar value.

*Costs*. A ControlProjection has four cost functions that determine how the ControlProjection computes the cost of
its control signal, all of which can be customized, and the first three of which can be enabled or disabled:

.. _ControlProjection_Cost_Functions:

* :keyword:`INTENSTITY_COST_FUNCTION`
    Calculates a cost based on the control signal :py:data:`intensity <ControlProjection.intensity>`.
    It can be any :class:`TransferFunction`, or any other function  that takes and returns a scalar value.
    The default is :class:`Exponential`.

* :keyword:`ADJUSTMENT_COST_FUNCTION`
    Calculates a cost based on a change in the control signal :py:data:`intensity <ControlProjection.intensity>`
    from its last value. It can be any :class:`TransferFunction`, or any other function that takes and returns a
    scalar value. The default is :py:class:`Function.Linear`.

* :keyword:`DURATION_COST_FUNCTION`
    Calculates an integral of the ControlProjection's :py:data:`cost <ControlProjection.cost>`.
    It can be any :class:`IntegratorFunction`, or any other function  that takes a list or array of two values and
    returns a scalar value. The default is :class:`Integrator`.

* :keyword:`COST_COMBINATION_FUNCTION`
    Combines the results of any cost functions that are enabled, and assigns the result as the ControlProjection's
    :py:data:`cost <ControlProjection.cost>`.  It can be any function that takes an array and returns a scalar value.
    The default is :py:class:`Reduce`.

An attribute is assigned for each component of the cost
(:py:data:`intensityCost <ControlProjection.intensityCost>`,
:py:data:`adjustmentCost <ControlProjection.adjustmentCost>`, and
:py:data:`durationCost <ControlProjection.durationCost>`),
and the total cost (:py:data:`cost <ControlProjection.cost>`.

.. _ControlProjection_Toggle_Costs:

*Toggling Cost Functions*.  Any of the cost functions (except the :keyword:`COST_COMBINATION_FUNCTION`) can be
enabled or disabled using the :py:meth:`toggle_cost_function <ControlProjection.toggle_cost_function>` method
to turn it :keyword:`ON` or :keyword:`OFF`.  If it is disabled, that component of the cost is not included in the
ControlProjection's :py:data:`cost <ControlProjection.cost>` attribute.  A cost function  can also be permanently
disabled for the ControlProjection by assigning :keyword:`None` to its argument in the constructor (or the
corresponding entry in its ``params`` dictionary).  If a cost function is permanently disabled for a ControlProjection,
it cannot be re-enabled using :py:meth:`toggle_cost_function <ControlProjection.toggle_cost_function>`.

*Additional Attributes*.  In addition to the intensity and cost attributes described above, a ControlProjection has
:py:data:`lastAllocation <ControlProjection.lastAllocation>` and
:py:data:`lastIntensity <ControlProjection.lastIntensity>` attributes that store the values associated with its
previous execution. Finally, it has an :py:data:`allocation_samples <ControlProjection.allocation_samples>` attribute,
that is a  list of used by :ref:`ControlMechanisms  <ControlMechanism>` for sampling different values of
:py:data:`allocation <ControlProjection.allocation>` for the ControlProjection, in order to adaptively adjust the
parameters that it controls (e.g., :doc:`EVCMechanism`). The default value is an array that ranges from
0.1 to 1.0 in steps of 0.1.

.. _ControlProjection_Execution:

Execution
---------

A ControlProjection uses its ``function`` to compute the :py:data:`intensity <ControlProjection.intensity>` of its
control signal, and its :ref:`cost functions <ControlProjection_Cost_Functions> use that to compute the its
:py:data:`cost <ControlProjection.cost>`.  The :py:data:`intensity <ControlProjection.intensity>` is assigned to the
ControlProjection's ``value`` attribute, which is used by the parmaterState to which it projects to modify the
corresponding parameter of the owner mechanism's function.  The :py:data:`cost <ControlProjection.cost>` is used by
the :doc:`EVCMechanism` to determine the ControlProjection's :py:data:`allocation <ControlProjection.allocation>`
in future executions.

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
        converts the ControlProjection's :py:data:`allocation <ControlProjection.allocation>` into its
        control signal :py:data:`intensity <ControlProjection.intensity>` (equal to its ``value``).

    intensity_cost_function : Optional[TransferFuntion] : default Exponential
        calculates a cost based on the control signal :py:data:`intensity <ControlProjection.intensity>`.
        It can be disabled permanently for the ControlProjection by assigning :keyword:`None`.

    adjustment_cost_function : Optiona[TransferFunction] : default Linear
        calculates a cost based on a change in the control signal :py:data:`intensity <ControlProjection.intensity>`
        from its last value. It can be disabled permanently for the ControlProjection by assigning :keyword:`None`.

    duration_cost_function : Optional[TransferFunction] : default Linear
        Calculates an integral of the ControlProjection's :py:data:`cost <ControlProjection.cost>`.
        It can be disabled permanently for the ControlProjection by assigning :keyword:`None`.

    cost_combination_function : function : default Reduce(operation=SUM)
        Combines the results of any cost functions that are enabled, and assigns the result to
        :py:data:`cost <ControlProjection.cost>`.

    allocation_samples : list : default :keyword:`DEFAULT_ALLOCATION_SAMPLES`
        Set of values used by ControlMechanisms that sample different allocation values in order to adaptively adjust
        the function of mechanisms in their systems.  The default value is an array that ranges from 0.1 to 1 in steps
        of 0.1.

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

    COMMENT:
      ControlSignal_General_Attributes
    COMMENT

    sender : OutputState of ControlProjection
        mechanism that provides the current :py:data:`allocation <ControlProjection.allocation>` for the
        ControlProjection.

    receiver : ParameterState of Mechanism
        parameterState for the parameter to be modified by ControlProjection.

    allocation : float : default: defaultControlAllocation
        value used as ``variable`` for ControlProjection's ``function`` to determine its control signal
        :py:data:`intensity <ControlProjection.intensity>`.

    allocationSamples : list : DEFAULT_SAMPLE_VALUES
        set of values used by ControlMechanisms that sample different allocation values in order to
        adaptively adjust the function of mechanisms in their systems.

        .. _ControlSignal_Function_Attributes:

    function : TransferFunction :  default Linear
        converts :py:data:`allocation <ControlProjection.allocation>` into `control signal
        :py:data:`intensity <ControlProjection.intensity>` that is provided as output to receiver of projection.

    intensityCostFunction : TransferFunction : default Exponential
        calculates "intensityCost`` from the curent value of :py:data:`intensity <ControlProjection.intensity>`.

    adjustmentCostFunction : TransferFunction : default Linear
        calculates :py:data:`adjustmentCost <ControlProjection.adjustmentCost>` based on the change in
        :py:data:`intensity <ControlProjection.intensity>` from its last value.

    durationCostFunction : IntegratorFunction : default Linear
        calculates an integral of the ControlProjection's :py:data:`cost <ControlProjection.cost>`.

    costCombinationFunction : function : default Reduce(operation=SUM)
        combines the results of any cost functions that are enabled, and assigns the result to
        :py:data:`cost <ControlProjection.cost>`.

    COMMENT:
        ControlSignal_State_Attributes:
    COMMENT

    value : float
        during initialization, assigned keyword string (either INITIALIZING or DEFERRED_INITIALIZATION);
        during execution, returns value of :py:data:`intensity <ControlProjection.intensity>`.

    intensity : float
        output of ``function``, used to determine controlled parameter of task.

    intensityCost : float
        cost associated with current :py:data:`intensity <ControlProjection.intensity>`.

    adjustmentCost : float
        cost associated with last change to :py:data:`intensity <ControlProjection.intensity>`.

    durationCost
        intregral of :py:data:`cost <ControlProjection.cost>`.

    cost : float
        current value of ControlProjection's :py:data:`cost <ControlProjection.cost>`;
        combined result of all cost functions that are enabled.

    COMMENT:
        ControlSignal_History_Attributes:
    COMMENT

    lastAllocation : float
        :py:data:`allocation <ControlProjection.allocation>` for last execution of the ControlProjection.

    lastIntensity : float
        :py:data:`intensity <ControlProjection.intensity>` for last execution of the ControlProjection.

        .. _ControlProjection_Cost_Functions:

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
            - creates projection-dedicated inputState, outputState and ControlSignalChannel in DefaultController
            - puts them in DefaultController's inputStates, outputStates, and ControlSignalChannels attributes
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

    def execute(self, variable=None, params=None, time_scale=None, context=None):
        """Adjust the control signal, based on the allocation value passed to it

        Computes new intensity and cost attributes from allocation
        Returns ControlSignalValuesTuple (intensity, totalCost)

        Use self.function to assign intensity
            - if ignoreIntensityFunction is set (for effiency, if the the execute method it is the identity function):
                ignore self.function
                pass allocation (input to controlSignal) along as its output
        Update cost

        :parameter allocation: (single item list, [0-1])
        :return: (intensity)
        """

        # store previous state
        self.lastAllocation = self.allocation
        self.lastIntensity = self.intensity
        self.last_cost = self.cost
        self.last_duration_cost = self.durationCost

        # update current intensity
        # FIX: IS THIS CORRECT?? OR SHOULD IT INDEED BE self.variable?
        # self.allocation = variable
        self.allocation = self.sender.value

        if self.ignoreIntensityFunction:
            # self.set_intensity(self.allocation)
            self.intensity = self.allocation
        else:
            self.intensity = self.function(self.allocation, params)
        intensity_change = self.intensity-self.lastIntensity

        if self.prefs.verbosePref:
            intensity_change_string = "no change"
            if intensity_change < 0:
                intensity_change_string = str(intensity_change)
            elif intensity_change > 0:
                intensity_change_string = "+" + str(intensity_change)
            if self.prefs.verbosePref:
                warnings.warn("\nIntensity: {0} [{1}] (for allocation {2})".format(self.intensity,
                                                                                   intensity_change_string,
                                                                                   self.allocation))
                warnings.warn("[Intensity function {0}]".format(["ignored", "used"][self.ignoreIntensityFunction]))


        # compute cost(s)
        # MODIFIED 11/16/16 NEW:
        new_cost = intensity_cost = adjustment_cost = duration_cost = 0

        if self.controlSignalCostOptions & ControlSignalCostOptions.INTENSITY_COST:
            intensity_cost = self.intensityCost = self.intensityCostFunction(self.intensity)
            if self.prefs.verbosePref:
                print("++ Used intensity cost")

        if self.controlSignalCostOptions & ControlSignalCostOptions.ADJUSTMENT_COST:
            adjustment_cost = self.adjustmentCost = self.adjustmentCostFunction(intensity_change)
            if self.prefs.verbosePref:
                print("++ Used adjustment cost")

        if self.controlSignalCostOptions & ControlSignalCostOptions.DURATION_COST:
            duration_cost = self.durationCost = self.durationCostFunction([self.last_duration_cost, new_cost])
            if self.prefs.verbosePref:
                print("++ Used duration cost")

        new_cost = self.costCombinationFunction([float(intensity_cost), adjustment_cost, duration_cost])

        if new_cost < 0:
            new_cost = 0
        self.cost = new_cost


        # Report new values to stdio
        if self.prefs.verbosePref:
            cost_change = new_cost - self.last_cost
            cost_change_string = "no change"
            if cost_change < 0:
                cost_change_string = str(cost_change)
            elif cost_change > 0:
                cost_change_string = "+" + str(cost_change)
            print("Cost: {0} [{1}])".format(self.cost, cost_change_string))

        #region Record controlSignal values in receiver mechanism log
        # Notes:
        # * Log controlSignals for ALL states of a given mechanism in the mechanism's log
        # * Log controlSignals for EACH state in a separate entry of the mechanism's log

        # Get receiver mechanism and state
        receiver_mech = self.receiver.owner
        receiver_state = self.receiver

        # Get logPref for mechanism
        log_pref = receiver_mech.prefs.logPref

        # Get context
        if not context:
            context = receiver_mech.name + " " + self.name + kwAssign
        else:
            context = context + SEPARATOR_BAR + self.name + kwAssign

        # If context is consistent with log_pref:
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and EXECUTING in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (EXECUTING in context))):
            # record info in log

# FIX: ENCODE ALL OF THIS AS 1D ARRAYS IN 2D PROJECTION VALUE, AND PASS TO .value FOR LOGGING
            receiver_mech.log.entries[receiver_state.name + " " +
                                      kpIntensity] = LogEntry(CurrentTime(), context, float(self.intensity))
            if not self.ignoreIntensityFunction:
                receiver_mech.log.entries[receiver_state.name + " " + kpAllocation] =     \
                    LogEntry(CurrentTime(), context, float(self.allocation))
                receiver_mech.log.entries[receiver_state.name + " " + kpIntensityCost] =  \
                    LogEntry(CurrentTime(), context, float(self.intensityCost))
                receiver_mech.log.entries[receiver_state.name + " " + kpAdjustmentCost] = \
                    LogEntry(CurrentTime(), context, float(self.adjustmentCost))
                receiver_mech.log.entries[receiver_state.name + " " + kpDurationCost] =   \
                    LogEntry(CurrentTime(), context, float(self.durationCost))
                receiver_mech.log.entries[receiver_state.name + " " + kpCost] =           \
                    LogEntry(CurrentTime(), context, float(self.cost))
    #endregion

        return self.intensity

    @property
    def allocationSamples(self):
        return self._allocation_samples

    @allocationSamples.setter
    def allocationSamples(self, samples):
        if isinstance(samples, (list, np.ndarray)):
            self._allocation_samples = list(samples)
            return
        if isinstance(samples, tuple):
            self._allocation_samples = samples
            sample_range = samples
        elif samples == AUTO:
            # THIS IS A STUB, TO BE REPLACED BY AN ACTUAL COMPUTATION OF THE ALLOCATION RANGE
            raise ControlProjectionError("AUTO not yet supported for {} param of ControlProjection; default will be used".
                                     format(ALLOCATION_SAMPLES))
        else:
            sample_range = DEFAULT_ALLOCATION_SAMPLES
        self._allocation_samples = []
        i = sample_range[0]
        while i < sample_range[1]:
            self._allocation_samples.append(i)
            i += sample_range[2]

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, new_value):
        try:
            old_value = self._intensity
        except AttributeError:
            old_value = 0
        self._intensity = new_value
        # if len(self.observers[kpIntensity]):
        #     for observer in self.observers[kpIntensity]:
        #         observer.observe_value_at_keypath(kpIntensity, old_value, new_value)

    def toggle_cost_function(self, cost_function_name, assignment=ON):
        """Enables/disables use of a cost function.

        ``cost_function_name`` should be a keyword (list under :ref:`Structure <ControlProjection_Structure>`).
        """

        if cost_function_name == INTENSITY_COST_FUNCTION:
            cost_option = ControlSignalCostOptions.INTENSITY_COST
        elif cost_function_name == DURATION_COST_FUNCTION:
            cost_option = ControlSignalCostOptions.DURATION_COST
        elif cost_function_name == ADJUSTMENT_COST_FUNCTION:
            cost_option = ControlSignalCostOptions.ADJUSTMENT_COST
        elif cost_function_name == COST_COMBINATION_FUNCTION:
            raise ControlProjectionError("{} cannot be disabled".format(COST_COMBINATION_FUNCTION))
        else:
            raise ControlProjectionError("toggle_cost_function: unrecognized cost function: {}".format(cost_function_name))

        if assignment:
            if not self.paramsCurrent[cost_function_name]:
                raise ControlProjectionError("Unable to toggle {} ON as function assignment is \'None\'".
                                         format(cost_function_name))
            self.controlSignalCostOptions |= cost_option
        else:
            self.controlSignalCostOptions &= ~cost_option

    # def set_intensity_cost(self, assignment=ON):
    #     if assignment:
    #         self.controlSignalCostOptions |= ControlSignalCostOptions.INTENSITY_COST
    #     else:
    #         self.controlSignalCostOptions &= ~ControlSignalCostOptions.INTENSITY_COST
    #
    # def set_adjustment_cost(self, assignment=ON):
    #     if assignment:
    #         self.controlSignalCostOptions |= ControlSignalCostOptions.ADJUSTMENT_COST
    #     else:
    #         self.controlSignalCostOptions &= ~ControlSignalCostOptions.ADJUSTMENT_COST
    #
    # def set_duration_cost(self, assignment=ON):
    #     if assignment:
    #         self.controlSignalCostOptions |= ControlSignalCostOptions.DURATION_COST
    #     else:
    #         self.controlSignalCostOptions &= ~ControlSignalCostOptions.DURATION_COST
    #
    def get_costs(self):
        """Return three-element list with the values of ``intensityCost``, ``adjustmentCost`` and ``durationCost``
        """
        return [self.intensityCost, self.adjustmentCost, self.durationCost]

    @property
    def value(self):
        if isinstance(self._value, str):
            return self._value
        else:
            return self._intensity

    @value.setter
    def value(self, assignment):
        self._value = assignment


# def RegisterControlSignal():
#     ProjectionRegistry(ControlProjection)
