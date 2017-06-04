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

A GatingProjection is a subclass of `Projection` that modulates the function of the `inputState 
<InputState>` or `outputState <OutputState>` of a `ProcessingMechanism`. It takes the value of an 
`outputState <OutputState> of another mechanism (e.g., usually a `GatingMechanism <GatingMechanism>`), and uses it 
to modulate the value of the state to which it projects.

.. _GatingProjection_Creation:

Creating a GatingProjection
----------------------------

A GatingProjection can be created using any of the standard ways to `create a projection <Projection_Creation>`,
or by including it in the specification of an inputState or outputState .  If a GatingProjection is created using its
constructor on its own, the `receiver <GatingProjection.receiver>` argument must be specified.  It can be specified
as a particular inputState or outputState of a mechanism, or simply as a mechanism.  In the latter case, 
the mechanism's `primary inputState <Mechanism_InputStates>` will be used. If the GatingProjection is included in an
inputState or outputState specification, that state will be assigned as the GatingProjection's 
`receiver <GatingProjection.receiver>`.  If GatingProjection's `sender <GatingProjection.sender>` is not
specified, the `sender <GatingProjection.sender>` is assigned to the outputState of a `DefaultGatingMechanism`.

.. _GatingProjection_Structure:

Structure
---------

A GatingProjection has the same structure as a `Projection`.  Its `sender <GatingProjection.sender>`
can be the outputState of any mechanism, but is generally a `GatingMechanism <GatingMechanism>`.  Its 
`receiver <GatingProjection.receiver>` is the `inputState <InputState>` or `outputState <OutputState>`  
of a `ProcessingMechanism`.  The `function <GatingProjection.function>` of a
GatingProjection is, by default, the identity function;  that is, it uses the :keyword:`value` of its `sender
<GatingProjection.sender>` to modulate the value of the state to which it projects.

.. _GatingProjection_Execution:

Execution
---------

A GatingProjection uses its `function <GatingProjection.function>` to assign its `value <GatingProjection.value>`
from the one received from its `sender <GatingProjection.sender>`.  This is used by the inputState or outputState to 
which the GatingProjection projects to modulate its own value.

.. note::
   The changes in an inputState or outputState's value in response to the execution of a GatingProjection are not 
   applied until the mechanism to which the state belongs is next executed; see :ref:`Lazy Evaluation` for an 
   explanation of "lazy" updating).

.. _GatingProjection_Class_Reference:


Class Reference
---------------

"""

# from PsyNeuLink.Components import DefaultGatingMechanism
from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection import ModulatoryProjection_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingMechanism import GatingMechanism


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

     Implements a projection that modulates the value of the inputState or outputState of a ProcessingMechanism.

    COMMENT:
        Description:
            The GatingProjection class is a type in the Projection category of Component.
            It implements a projection to the inputState or outputState of a mechanism that modulates the value of 
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
            + paramNames = paramClassDefaults.keys()
    COMMENT


    Arguments
    ---------

    sender : Optional[Mechanism or GatingSignal]
        specifies the source of the input for the GatingProjection;  usually an `outputState <OutputState>` of a
        `GatingMechanism <GatingMechanism>`.  If it is not specified, an outputState of the `DefaultGatingMechanism` 
        for the system to which the receiver belongs will be assigned.

    receiver : Optional[Mechanism or ParameterState]
        specifies the inputState or outputState to which the GatingProjection projects.  This must be specified,
        or be able to be determined by the context in which the GatingProjection is created or assigned.

    function : TransferFunction : default Linear
        specifies the function used to convert the :keyword:`value` of the GatingProjection's
        `sender <GatingProjection.sender>`  to its own `value <GatingProjection.value>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
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

    sender : OutputState of GatingMechanism
        mechanism that provides the current input for the GatingProjection (usually a
        `GatingMechanism <GatingMechanism>`).

    receiver : InputState or OutputState of a ProcessingMechanism
        `inputState <InputState>` or `outputState <OutputState>` to which the GatingProjection projects.

    gatting_signal : 1d np.array
        the input to the GatingProjection; same as the :keyword:`value` of the `sender <GatingProjection.sender>`.

    value : float
        during initialization, assigned a keyword string (either `INITIALIZING` or `DEFERRED_INITIALIZATION`);
        during execution, is assigned the current value of the GatingProjection.

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

    variableClassDefault = 0.0

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_SENDER: GatingMechanism,
        PROJECTION_SENDER_VALUE: defaultGatingPolicy})

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 function=Linear,
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
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = self
            self.init_args['name'] = name
            # Delete this as it has breen moved to params dict (so it will not be passed to Projection.__init__)
            del self.init_args[GATING_SIGNAL_PARAMS]

            # Flag for deferred initialization
            self.value = DEFERRED_INITIALIZATION
            return

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
                                  "to a mechanism {0} in pathway list".
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
    def gating_policy(self):
        return self.sender.value
