# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  GatingProjection ******************************************************

"""
.. _GatingProjection_Overview:

Overview
--------

A GatingProjection is a type of `ModulatoryProjection` that projects to the `InputState` or `OutputState` of a
`Mechanism`. It takes the value of a `GatingSignal` of a `GatingMechanism`, and uses it to modulate the `value
<State_Base.value>` of the State to which it projects.

.. _GatingProjection_Creation:

Creating a GatingProjection
----------------------------

A GatingProjection can be created using any of the standard ways to `create a projection <Projection_Creation>`,
or by including it in the specification of an `InputState <InputState_Projections>` or
`OutputState <OutputState_Projections>` .  If a GatingProjection is created explicitly (using its constructor),
its **receiver** argument can be specified as a particular InputState or OutputState of a designated `Mechanism`,
or simply as the Mechanism.  In the latter case, the Mechanism's `primary InputState <InputState_Primary>` will be
used. If the GatingProjection is included in an InputState or OutputState specification, that State will be assigned
as the GatingProjection's `receiver <GatingProjection.receiver>`. If the **sender** and/or **receiver** arguments are
not specified, its initialization is `deferred  <GatingProjection_Deferred_Initialization>`.


.. _GatingProjection_Deferred_Initialization:

Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~

When a GatingProjection is created, its full initialization is `deferred <Component_Deferred_Init>` until its
`sender <ControlProjection.sender>` and `receiver <ControlProjection.receiver>` have been fully specified.  This allows
a GatingProjection to be created before its `sender` and/or `receiver` have been created (e.g., before them in a
script), by calling its constructor without specifying its **sender** or **receiver** arguments. However, for the
GatingProjection to be operational, initialization must be completed by calling its `deferred_init` method.  This is
not necessary if the State(s) to be gated are specified in the **gating_signals** argument of a `GatingMechanism
<GatingMechanism_Specifying_Gating>`, in which case deferred initialization is completed automatically by the
GatingMechanism when it is created.

.. _GatingProjection_Structure:

Structure
---------

The `sender <GatingProjection.sender>` of a GatingProjection is a `GatingSignal` of a `GatingMechanism`.  The `value
<GatingSignal.value>` of the `sender <GatingProjection.sender>` is used by the GatingProjection as its
`variable <GatingProjection.variable>`;  this is also assigned to its `gating_signal
<GatingProjection.gating_signal>` attribute, and serves as the input to the GatingProjection's `function
<GatingProjection.function>`.  The default `function <GatingProjection.function>` for a
GatingProjection is an identity function (`Linear` with **slope**\\ =1 and **intercept**\\ =0);  that is,
it simply conveys the value of its `gating_signal <GatingProjection.gating_signal>` to its `receiver
<GatingProjection.receiver>`, for use in modifying the `value <State_Base.value>` of the State that it gates. Its
`receiver <GatingProjection.receiver>` is the `InputState` or `OutputState` of a `Mechanism`.

.. _GatingProjection_Execution:

Execution
---------

A GatingProjection cannot be executed directly.  It is executed when the `InputState` or `OutputState` to which it
projects is updated.  Note that this only occurs when the `Mechanism` to which the `State` belongs is executed (see
:ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating). When a GatingProjection is executed,
its `function <GatingProjection.function>` gets the `gating_signal <GatingProjection.gating_signal>` from its `sender
<GatingProjection.sender>` and conveys that to its `receiver <GatingProjection.receiver>`.  This is used by the
`receiver <GatingProjection.receiver>` to modify the `value <State_Base.value>` of the State gated by the
GatingProjection (see `ModulatorySignal_Modulation`, `InputState Execution <InputState_Execution>` and
`OutputState Execution <OutputState_Execution>` for how modulation operates and how this applies to a InputStates and
OutputStates).

.. note::
   The changes in an InputState or OutputState's `value <State_Base.value >` in response to the execution of a
   GatingProjection are not applied until the Mechanism to which the State belongs is next executed;
   see :ref:`Lazy Evaluation` for an explanation of "lazy" updating).

.. _GatingProjection_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc


from PsyNeuLink.Components.Component import InitStatus, parameter_keywords
from PsyNeuLink.Components.Functions.Function import FunctionOutputType, Linear
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingMechanism import GatingMechanism
from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection import ModulatoryProjection_Base
from PsyNeuLink.Components.Projections.Projection import ProjectionError, Projection_Base, projection_keywords
from PsyNeuLink.Components.ShellClasses import Mechanism, Process
from PsyNeuLink.Globals.Defaults import defaultGatingPolicy
from PsyNeuLink.Globals.Keywords import FUNCTION_OUTPUT_TYPE, GATING, GATING_MECHANISM, GATING_PROJECTION, INITIALIZING, PROJECTION_SENDER, PROJECTION_SENDER_VALUE
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Scheduling.TimeScale import CentralClock

parameter_keywords.update({GATING_PROJECTION, GATING})
projection_keywords.update({GATING_PROJECTION, GATING})
GATING_SIGNAL_PARAMS = 'gating_signal_params'

class GatingProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class GatingProjection(ModulatoryProjection_Base):
    """
    GatingProjection(  \
     sender=None,      \
     receiver=None,    \
     function=Linear   \
     params=None,      \
     name=None,        \
     prefs=None)

    Subclass of `ModulatoryProjection` that modulates the value of an `InputState` or `OutputState`.

    COMMENT:
        Description:
            The GatingProjection class is a type in the Projection category of Component.
            It implements a projection to the InputState or OutputState of a Mechanism that modulates the value of
            that state
            It:
               - takes a scalar as its input (sometimes referred to as a "gating signal")
               - uses its `function` to compute its value
               - its value is used to modulate the value of the state to which it projects

        ** MOVE:
        ProjectionRegistry:
            All GatingProjections are registered in ProjectionRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

        Class attributes:
            + color (value): for use in interface design
            + classPreference (PreferenceSet): GatingProjectionPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + paramClassDefaults:
                FUNCTION:Linear,
                FUNCTION_PARAMS:{SLOPE: 1, INTERCEPT: 0},  # Note: this implements identity function
                PROJECTION_SENDER: DefaultGatingMechanism, # GatingProjection (assigned to class ref in __init__ module)
                PROJECTION_SENDER_VALUE: [defaultGatingSignal]
    COMMENT


    Arguments
    ---------

    sender : Optional[GatingMechanism or GatingSignal]
        specifies the source of the `gating_signal <GatingProjection.gating_signal>` for the GatingProjection;
        if it is not specified and cannot be `inferred from context <GatingProjection_Creation>` , initialization is
        `deferred <GatingProjection_Deferred_Initialization>`.

    receiver : Optional[Mechanism, InputState or OutputState]
        specifies the `InputState` or `OutputState` to which the GatingProjection projects; if it is not specified,
        and cannot be `inferred from context <GatingProjection_Creation>`, initialization is `deferred
        <GatingProjection_Deferred_Initialization>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies the function used to convert the `gating_signal <GatingProjection.gating_signal>` to the
        GatingProjection's `value <GatingProjection.value>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the projection, its `function <GatingProjection.function>`, and/or a custom function and its parameters.
        Values specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default GatingProjection-<index>
        a string used for the name of the GatingProjection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Projection.classPreferences]
        the `PreferenceSet` for the GatingProjection.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

    componentType : GATING_PROJECTION

    sender : GatingSignal
        source of the `gating_signal <GatingProjection.gating_signal>`.

    receiver : InputState or OutputState of a Mechanism
        `InputState` or `OutputState` to which the GatingProjection projects.

    variable : 2d np.array
        same as `gating_signal <GatingProjection.gating_signal>`.

    gating_signal : 1d np.array
        the `value <GatingSignal.value>` of the GatingProjection's `sender <GatingProjection.sender>`.

    function : Function
        assigns the `gating_signal` received from the `sender <GatingProjection.sender>` to the
        GatingProjection's `value <GatingProjection.value>`; the default is an identity function.

    value : float
        the value used to modify the `value <State_Base.value>` of the `InputState` or `OutputState` gated by the
        GatingProjection (see `ModulatorySignal_Modulation`, `InputState Execution <InputState_Execution>`, and
        `OutputState Execution <OutputState_Execution>` for how modulation operates and how this applies to InputStates
        and OutputStates).

    name : str : default GatingProjection-<index>
        the name of the GatingProjection.
        Specified in the **name** argument of the constructor for the projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for projection.
        Specified in the **prefs** argument of the constructor for the projection;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    """

    color = 0

    componentType = GATING_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    class ClassDefaults(ModulatoryProjection_Base.ClassDefaults):
        variable = 0.0

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_SENDER: GatingMechanism,
        PROJECTION_SENDER_VALUE: defaultGatingPolicy})

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 function=Linear(params={FUNCTION_OUTPUT_TYPE:FunctionOutputType.RAW_NUMBER}),
                 # function=Linear,
                 gating_signal_params:tc.optional(dict)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  gating_signal_params=gating_signal_params,
                                                  params=params)

        # If receiver has not been assigned, defer init to State.instantiate_projection_to_state()
        if sender is None or receiver is None:
            # Flag for deferred initialization
            self.init_status = InitStatus.DEFERRED_INITIALIZATION

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        super().__init__(sender=sender,
                         receiver=receiver,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _instantiate_sender(self, params=None, context=None):
        """Check that sender is not a process and that, if specified as a Mechanism, it is a GatingMechanism
        """

        # A Process can't be the sender of a GatingProjection
        if isinstance(self.sender, Process):
            raise ProjectionError("PROGRAM ERROR: attempt to add a {} from a Process {0} "
                                  "to a Mechanism {0} in pathway list".
                                  format(GATING_PROJECTION, self.name, self.sender.name))

        # If sender is specified as a Mechanism, validate that it is a GatingMechanism
        if isinstance(self.sender, Mechanism):
            if not isinstance(self.sender, GatingMechanism):
                raise GatingProjectionError("Mechanism specified as sender for {} ({}) must be a {} (but it is a {})".
                                    format(GATING_MECHANISM,self.name, self.sender.name,self.sender.__class__.__name__))

        # Call super to instantiate sender
        super()._instantiate_sender(context=context)

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if INITIALIZING in context:
            from PsyNeuLink.Components.States.InputState import InputState
            from PsyNeuLink.Components.States.OutputState import OutputState
            if not isinstance(self.receiver, (InputState, OutputState, Mechanism)):
                raise GatingProjectionError("Receiver specified for {} {} is not a "
                                            "Mechanism, InputState or OutputState".
                                            format(self.receiver, self.name))

    def _instantiate_receiver(self, context=None):
        """Assign state if receiver is Mechanism, and match output to param being modulated
        """
        # If receiver specification was a Mechanism, re-assign to the mechanism's primary inputState
        if isinstance(self.receiver, Mechanism):
            # If Mechanism is specified as receiver, assign GatingProjection to primary inputState as the default
            self.receiver = self.receiver.input_states[0]

        # # Match type of GatingProjection.value to type to the parameter being modulated
        # modulated_param = self.sender.modulation
        # function = self.receiver.function_object
        # function_param = function.params[modulated_param]
        # function_param_value = function.params[function_param]
        # gating_projection_function = self.function.__self__
        # gating_projection_function.functionOutputType = type(function_param_value)
        # # ASSIGN FUNCTION TYPE TO FUNCTION HERE

        super()._instantiate_receiver(context=context)

    def execute(self, params=None, clock=CentralClock, time_scale=None, context=None):
    # def execute(self, params=None, clock=CentralClock, time_scale=TimeScale.TRIAL, context=None):
        self.variable = self.sender.value
        self.value = self.function(variable=self.variable, params=params, time_scale=time_scale, context=context)
        return self.value

    @property
    def gating_signal(self):
        return self.sender.value
