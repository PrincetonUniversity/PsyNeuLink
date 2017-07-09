# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  GatingMechanism ************************************************

"""
Overview
--------

A GatingMechanism is an `AdaptiveMechanism` that modulates the value of the InputState(s) and/or OutputState(s) of
one or more `ProcessingMechanisms`.   Its function takes a value
COMMENT:
    ??FROM WHERE?
COMMENT
and uses that to calculate a `gating_policy`:  a list of `gating <LINK>` values, one for each of states that it 
gates.  Each of these values is assigned as the value of a `GatingSignal` (a subclass of `OutputState`) in the
GatingMechanism, and used by an associated `GatingProjection` to modulate the value of the state to which it projects.  
A GatingMechanism can regulate only the parameters of Mechanisms in the `System` to which it belongs.
COMMENT: TBI
The gating components of a System can be displayed using the System's
`show_graph` method with its **show_gating** argument assigned as :keyword:``True`.  
COMMENT
The gating components of a System are executed after all `Proces singMechanisms <ProcessingMechanism>`,
`LearningMechanisms <LearningMechanism>`, and  `ControlMechanisms <ControlMechanism>` in that System have been executed.


.. _GatingMechanism_Creation:

Creating A GatingMechanism
---------------------------

GatingMechanisms can be created using the standard Python method of calling the constructor for the desired type.
A GatingMechanism is also created automatically if `gating is specified <GatingMechanism_Specifying_Gating>` for an 
InputState or OutputState, in which case a `GatingProjection` is also automatically created that projects
from the GatingMechanism to the specified state.

.. _GatingMechanism_Specifying_Gating:

Specifying gating
~~~~~~~~~~~~~~~~~

GatingMechanisms are used to modulate the value of an `InputState` or `OutputState`.
An InputState or OutputState can be specified for gating by assigning it a `GatingProjection` in the
**input_states** or **output_states** arguments of the constructor for the Mechanism to which it belongs
(see `Mechanism_States <LINK>`).  The InputStates and OutputStates to be gated by a GatingMechanism can also be
specified in the  **gating_signals**  argument of the constructor for a GatingMechanism.  The **gating_signals** 
argument must be a list, each item of which must refer to one or more states to be gated by that GatingSignal.  
The specification for each item in the list can use any of the forms used to 
`specify a GatingSignal <GatingSignal_Specification>`.

.. _GatingMechanism_GatingSignals:

GatingSignals
^^^^^^^^^^^^^

A `GatingSignal` is created for each item listed in **gating_signals**, and all of the GatingSignals for a  
GatingMechanism are listed in its `gating_signals <GatingMechanism.gating_signals>` attribute.  Each GatingSignal is 
assigned one or more `GatingProjections <GatingProjection>` to the InputState(s) and/or OutputState(s) it gates.
GatingSignals are a type of `OutputState`, and so they are also listed in the GatingMechanism's 
`output_states <GatingMechanism.outut_states>` attribute.  

.. _GatingMechanism_Modulation:

Modulation
^^^^^^^^^^

Each GatingMechanism has a `modulation <GatingSignal.modulation>` attribute, that provides a default for the way
in which its GatingSignals modulate the value of the states they gate 
(see `modulation <ModulatoryProjection.modulation>` for an explanation of how this attribute is specified and used to 
modulate the value of a state).  Each GatingSignal uses this value, unless its value is 
`individually specified <GatingSignal_Modulation>`. 

.. _GatingMechanism_Execution:

Execution
---------

A GatingMechanism executes in the same way as a ProcessingMechanism, based on its place in the System's
`graph <System.graph>`.  Because GatingProjections are likely to introduce cycles (loops) in the graph,
the effects of a GatingMechanism and its projections will generally not be applied in the first `TRIAL` (see
`initialization <System_Execution_Input_And_Initialization>` for a description of how to configure the initialization
of feedback loops in a System).  When executed, a GatingMechanism uses its input to determine the value of its
`GatingSignals <GatingSignal>` and their corresponding `GatingProjections <GatingProjection>`.  In the subsequent 
`TRIAL`, each GatingProjection's value is used by the State to which it projects to modulate the `value <State.value>`
of that State.

When a GatingMechanism executes, the value of each item in its `gating_policy` is assigned as the value of each of
the corresponding GatingSignals in its `gating_signals` attribute.  These, in turn, are used by their associated
`GatingProjections` to modulate the value of the state to which they project.  This is done by assigning the
GatingSignal's value to a parameter of the state's function, as specified by the GatingSignal's `modulation` 
parameter (see `GatingSignal_Execution` for details). 

.. note::
   A state that receives a `GatingProjection` does not update its value until its owner Mechanism executes
   (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a GatingMechanism 
   has executed, a state that it gates will not assume its new value until the state's owner has executed.

.. _GatingMechanism_Class_Reference:

Class Reference
---------------

"""

# IMPLEMENTATION NOTE: COPIED FROM DefaultProcessingMechanism;
#                      ADD IN GENERIC CONTROL STUFF FROM DefaultGatingMechanism

from PsyNeuLink.Components.Functions.Function import ModulationParam, _is_modulation_param
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Projections.Projection import _validate_receiver
from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Components.States.ModulatorySignals.GatingSignal import GatingSignal, _parse_gating_signal_spec
from PsyNeuLink.Components.States.State import State_Base, _instantiate_state

GatingMechanismRegistry = {}


class GatingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class GatingMechanism(AdaptiveMechanism_Base):
    """
    GatingMechanism_Base(                           \
        default_input_value=None,                   \
        gating_signals=None                         \
        modulation=ModulationParam.MULTIPLICATIVE   \
        function=Linear,                            \
        params=None,                                \
        name=None,                                  \
        prefs=None)

    Abstract class for GatingMechanism.

    .. note::
       GatingMechanisms should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the constructor for a :doc:`subclass <GatingMechanism>`.

    COMMENT:
        Description:
            # VERIFY:
            Protocol for instantiating unassigned GatingProjections (i.e., w/o a sender specified):
               If sender is not specified for a GatingProjection (e.g., in an InputState or OutputState tuple spec)
                   it is flagged for deferred_init() in its __init__ method
               When the next GatingMechanism is instantiated, if its params[MAKE_DEFAULT_GATING_MECHANISM] == True, its
                   _take_over_as_default_gating_mechanism method is called in _instantiate_attributes_after_function;
                   it then iterates through all of the InputStates and OutputStates of all of the Mechanisms in its
                   System, identifies ones without a sender specified, calls its deferred_init() method,
                   instantiates a GatingSignal for it, and assigns it as the GatingProjection's sender.

        Class attributes:
            + componentType (str): System Default Mechanism
            + paramClassDefaults (dict):
                + FUNCTION: Linear
                + FUNCTION_PARAMS:{SLOPE:1, INTERCEPT:0}
    COMMENT

    Arguments
    ---------

    default_gating_policy : value, list or np.ndarray : :py:data:`defaultGatingPolicy <LINK]>`
        the default value for each of the GatingMechanism's GatingSignals;
        its length must equal the number of items specified in the **gating_signals** arg.

    gating_signals : List[InputState or OutputState, Mechanism, tuple[str, Mechanism], or dict]
        specifies the InputStates and/or OutputStates to be gated by the GatingMechanism;
        the number of items must equal the length of the **default_gating_policy** arg;
        if a Mechanism is specified, its `primary InputState <Mechanism_InputStates>` will be used
        (see `gating_signals <GatingMechanism.gating_signals>` for details).
        
    modulation : ModulationParam : ModulationParam.MULTIPLICATIVE
        specifies the default form of modulation used by the GatingMechanism's GatingSignals, unless
        they are `individually specified <GatingSignal_Specification>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to transform input to `gating_policy` to value;  the default is an identity function
        that simply assigns the GatingMechanism's input to its `value`.
        
    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default ControlMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for the Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    gating_signals : List[GatingSignal]
        list of `GatingSignals <ControlSignals>` for the GatingMechanism, each of which sends a `GatingProjection`
        to the `InputState` or `OutputState` that it gates (same as GatingMechanism's
        `output_states <Mechanism.output_states>` attribute).

    gating_projections : List[GatingProjection]
        list of `GatingProjections <GatingProjection>`, one for each `GatingSignal` in `gating_signals`.

    gating_policy : 2d np.array
        each item is the value assigned to the corresponding GatingSignal listed in `gating_signals`;
        (same as the GatingMechanism's `value <Mechanism.value>` attribute).
        Default is a single item used by all of the `gating_signals`. 

    modulation : ModulationParam
        the default form of modulation used by the GatingMechanism's GatingSignals, unless they are 
        `individually specified <GatingSignal_Specification>`.
        
    """

    componentType = "GatingMechanism"

    initMethod = INIT__EXECUTE__METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'GatingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # This must be a list, as there may be more than one (e.g., one per GATING_SIGNAL)
    variableClassDefault = defaultGatingPolicy

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({GATING_PROJECTIONS: None})

    @tc.typecheck
    def __init__(self,
                 default_gating_policy=None,
                 size=None,
                 function = Linear(slope=1, intercept=0),
                 gating_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # self.system = None

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gating_signals=gating_signals,
                                                  function=function,
                                                  params=params)

        super().__init__(variable=default_gating_policy,
                         size=size,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate items in the GATING_SIGNALS param (**gating_signals** argument of constructor)

        Check that GATING_SIGNALS is a list, and that every item is valid state_spec
        """

        super(GatingMechanism, self)._validate_params(request_set=request_set,
                                                      target_set=target_set,
                                                      context=context)

        if GATING_SIGNALS in target_set and target_set[GATING_SIGNALS]:
            
            if not isinstance(target_set[GATING_SIGNALS], list):
                raise GatingMechanismError("{} arg of {} must be list".
                                           format(GATING_SIGNAL, self.name))

            for spec in target_set[GATING_SIGNALS]:
                _parse_gating_signal_spec(self, spec)

    def _instantiate_output_states(self, context=None):

        # Create registry for GatingSignals (to manage names)
        from PsyNeuLink.Globals.Registry import register_category
        register_category(entry=GatingSignal,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        self.gating_signal_specs = self.gating_signals

        if self.gating_signal_specs:
            for gating_signal in self.gating_signal_specs:
                self._instantiate_gating_signal(gating_signal=gating_signal, context=context)
                TEST = True

        # IMPLEMENTATION NOTE:  Don't want to call this because it instantiates undesired default OutputState
        # super()._instantiate_output_states(context=context)

    def _instantiate_gating_signal(self, gating_signal=None, context=None):
        """Instantiate GatingSignal OutputState and assign (if specified) or instantiate GatingProjection

        # Extends gating_policy and to accommodate instantiated projection

        Notes:
        * gating_signal arg can be a:
            - GatingSignal object;
            - GatingProjection;
            - InputState or OutputState;
            - params dict, from _take_over_as_default_gating_mechanism(), containing a GatingProjection;
            - tuple (state_name, Mechanism), from gating_signals arg of constructor;
                    [NOTE: this is a convenience format;
                           it precludes specification of GatingSignal params (e.g., MODULATION_OPERARATION)]
            - GatingSignal specification dictionary, from gating_signals arg of constructor
                    [NOTE: this must have at least NAME:str (state name) and MECHANISM:Mechanism entries; 
                           it can also include a PARAMS entry with a params dict containing GatingSignal params] 
        * State._parse_state_spec() is used to parse gating_signal arg
        * params are expected to be for (i.e., to be passed to) GatingSignal;
        * wait to instantiate deferred_init() projections until after GatingSignal is instantiated,
            so that correct OutputState can be assigned as its sender;
        * index of OutputState is incremented based on number of GatingSignals already instantiated;
            this means that the GatingMechanism's function must return as many items as it has GatingSignals,
            with each item of the function's value used by a corresponding GatingSignal.
            Note: multiple GatingProjections can be assigned to the same GatingSignal to achieve "divergent gating"
                  (that is, gating of many states with a single value -- e.g., LC)

        Returns GatingSignal (OutputState)
        """

        from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

        # EXTEND gating_policy TO ACCOMMODATE NEW GatingSignal -------------------------------------------------
        #        also used to determine constraint on GatingSignal value

        if not hasattr(self, GATING_POLICY) or self.gating_policy is None:
            self.gating_policy = np.array(defaultGatingPolicy)
        else:
            self.gating_policy = np.append(self.gating_policy, defaultGatingPolicy)

        # GET index FOR GatingSignal OutputState
        try:
            # If there are as many OutputStates as items in self.value, assume that each
            if len(self.gating_signals) == len(self.value):
                output_state_index = len(self.gating_signals) - 1
            else:
                output_state_index = 0
        except (AttributeError, TypeError):
            output_state_index = 0


        # PARSE gating_signal SPECIFICATION -----------------------------------------------------------------------

        gating_projections = []
        gating_signal_params = {}

        gating_signal_spec = _parse_gating_signal_spec(owner=self, state_spec=gating_signal)

        # Specification is a GatingSignal (either passed in directly, or parsed from tuple above)
        if isinstance(gating_signal_spec, GatingSignal):
            gating_signal = gating_signal_spec[GATING_SIGNAL]
            # Deferred Initialization, so assign owner, name, and initialize
            if gating_signal.value is DEFERRED_INITIALIZATION:
                # FIX 5/23/17:  IMPLEMENT DEFERRED_INITIALIZATION FOR GatingSignal
                # CALL DEFERRED INIT WITH SELF AS OWNER ??AND NAME FROM gating_signal_dict?? (OR WAS IT SPECIFIED)
                # OR ASSIGN NAME IF IT IS DEFAULT, USING GATING_SIGNAL_DICT??
                pass
            elif not gating_signal_spec.owner is self:
                raise GatingMechanismError("Attempt to assign GatingSignal to {} ({}) that is already owned by {}".
                                            format(self.name, gating_signal_spec.name, gating_signal_spec.owner.name))
            gating_signal_name = gating_signal.name
            gating_projections = gating_signal.efferents

        # Instantiate OutputState for GatingSignal
        else:
            gating_signal_name = gating_signal_spec[NAME]
            # FIX: ??CALL REGISTRY FOR NAME HERE (AS FOR OUTPUTSTATE IN MECHANISM?? -
            # FIX:  OR IS THIS DONE AUTOMATICALLY IN _instantiate_state??)
            if gating_signal_name in [gs.name for gs in self.gating_signals if isinstance(gs, GatingSignal)]:
                gating_signal_name = gating_signal_name + '-' + repr(len(self.gating_signals))

            # from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingSignal import GatingSignal
            # from PsyNeuLink.Components.States.State import _instantiate_state

            # Get constraint for OutputState's value
            #    - get GatingMechanism's value
            self._update_value(context=context)
            # - get constraint for OutputState's value
            output_state_constraint_value = self.gating_policy[output_state_index]

            # gating_signal_params.update({GATED_STATE:state_name})
            gating_signal_params.update(gating_signal_spec[PARAMS])

            # FIX 5/23/17: CALL super()_instantiate_output_states ??
            # FIX:         OR AGGREGATE ALL GatingSignals AND SEND AS LIST (AS FOR input_states IN ObjectiveMechanism)
            gating_signal = _instantiate_state(owner=self,
                                               state_type=GatingSignal,
                                               state_name=gating_signal_name,
                                               state_spec=defaultGatingPolicy,
                                               state_params=gating_signal_params,
                                               constraint_value=output_state_constraint_value,
                                               constraint_value_name='Default control allocation',
                                               context=context)

        # VALIDATE OR INSTANTIATE GatingProjection(s) TO GatingSignal  -------------------------------------------

        # Validate gating_projections (if specified)
        if gating_projections:
            for gating_projection in gating_projections:
                if not isinstance(gating_projection, GatingProjection):
                    raise GatingMechanismError("PROGRAM ERROR: Attempt to assign {}, "
                                                      "that is not a GatingProjection, to GatingSignal of {}".
                                                      format(gating_projection, self.name))
                _validate_receiver(self, gating_projection, Mechanism, GATING_SIGNAL, context=context)
                state = gating_projection.receiver
                if gating_projection.value is DEFERRED_INITIALIZATION:
                    gating_projection.init_args['sender']=gating_signal
                    if gating_projection.init_args['name'] is None:
                        gating_projection.init_args['name'] = GATING_PROJECTION + \
                                                              ' for ' + state.name + ' of ' + state.owner.name
                    gating_projection._deferred_init()
                else:
                    gating_projection.sender = gating_signal

        # Instantiate GatingProjection
        else:
            for state in gating_signal_spec[STATES]:
                # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
                gating_projection = GatingProjection(sender=gating_signal,
                                                     receiver=state,
                                                     name=GATING_PROJECTION + gating_signal_name)
                # Add gating_projection to GatingSignal.efferents
                # gating_signal.efferents.append(gating_projection)
                gating_projections.append(gating_projection)

        # Add GatingProjections to GatingMechanism's list of GatingProjections
        for gating_projection in gating_projections:
            try:
                self.gating_projections.append(gating_projection)
            except AttributeError:
                self.gating_projections = [gating_projection]

        # FIX: CONSIDER OVERRIDING output_states PROPERTY WITH ASSIGNMENT TO gating_signals
        # UPDATE output_states
        try:
            self.output_states[gating_signal.name] = gating_signal
        except (AttributeError, TypeError):
            self.output_states = ContentAddressableList(component_type=State_Base,
                                                        list=[gating_signal],
                                                        name=self.name + '.output_states')
        # Add index assignment to OutputState
        gating_signal.index = output_state_index
        # (Re-)assign control_signals attribute to output_states
        self._gating_signals = self.output_states

        return gating_signal

    def _instantiate_attributes_after_function(self, context=None):
        """Take over as default GatingMechanism (if specified) and implement any specified GatingProjections
        """

        super()._instantiate_attributes_after_function(context=context)

        if MAKE_DEFAULT_GATING_MECHANISM in self.paramsCurrent:
            if self.paramsCurrent[MAKE_DEFAULT_GATING_MECHANISM]:
                self._take_over_as_default_gating_mechanism(context=context)

        # FIX: 5/23/17 CONSOLIDATE/SIMPLIFY THIS RE: gating_signal ARG??  USE OF STATE_PROJECTIONS, ETC.
        # FIX:         ?? WHERE WOULD GATING_PROJECTIONS HAVE BEEN SPECIFIED IN paramsCURRENT??
        # FIX:         DOCUMENT THAT VALUE OF GATING ENTRY CAN BE A PROJECTION
        # FIX:         RE-WRITE parse_state_spec TO TAKE TUPLE THAT SPECIFIES (PARAM VALUE, GATING SIGNAL)
        #                       RATHER THAN (PARAM VALUE, GATING PROJECTION)
        # FIX: NOT CLEAR THIS IS GETTING USED AT ALL; ALSO, ??REDUNDANT WITH CALL IN _instantiate_output_states
        # If GatingProjections were specified, implement them
        if GATING_PROJECTIONS in self.paramsCurrent:
            if self.paramsCurrent[GATING_PROJECTIONS]:
                for key, projection in self.paramsCurrent[GATING_PROJECTIONS].items():
                    self._instantiate_gating_projection(projection, context=self.name)

    def _take_over_as_default_gating_mechanism(self, context=None):

        # FIX 5/23/17: INTEGRATE THIS WITH ASSIGNMENT OF gating_signals
        # FIX:         (E.G., CHECK IF SPECIFIED GatingSignal ALREADY EXISTS)
        # Check the input_states and output_states of the System's Mechanisms
        #    for any GatingProjections with deferred_init()
        for mech in self.system.mechanisms:
            for state in mech._input_states + mech._output_states:
                for projection in state.mod_afferents:
                    # If projection was deferred for init, initialize it now and instantiate for self
                    if projection.value is DEFERRED_INITIALIZATION and projection.init_args['sender'] is None:
                        # FIX 5/23/17: MODIFY THIS WHEN (param, GatingProjection) tuple
                        # FIX:         IS REPLACED WITH (param, GatingSignal) tuple
                        # Add projection itself to any params specified in the GatingProjection for the GatingSignal
                        #    (cached in the GatingProjection's gating_signal attrib)
                        gating_signal_specs = projection.gating_signal or {}
                        gating_signal_specs.update({GATING_SIGNAL_SPECS: [projection]})
                        self._instantiate_gating_signal(gating_signal_specs, context=context)

    # IMPLEMENTATION NOTE: This is needed if GatingMechanism is added to a System but does not have any afferents
    #                      (including from ProcessInputState or SystemInputState)
    #                      and therefore variable = None
    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Updates GatingSignals based on inputs
        """

        if variable is None or variable[0] is None:
            variable = self.variableClassDefault

        return super()._execute(variable=variable,
                                runtime_params=runtime_params,
                                clock=clock,
                                time_scale=time_scale,
                                context=context)
        # gating_policy = self.function(variable=variable,
        #                               function_params=function_params,
        #                               time_scale=time_scale,
        #                               context=context)
        # return gating_policy


    def show(self):

        print ("\n---------------------------------------------------------")

        print ("\n{0}".format(self.name))
        print ("\n\tGating the following Mechanism InputStates and/or OutputStates:".format(self.name))
        # Sort for consistency of output:
        state_names_sorted = sorted(self.output_states.keys())
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].efferents:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))
        print ("\n---------------------------------------------------------")


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _add_gating_mechanism_to_system(owner:GatingMechanism):

    if owner.gating_signals:
        for gating_signal in owner.gating_signals:
            for mech in [proj.receiver.owner for proj in gating_signal.efferents]:
                for system in mech.systems:
                    if owner not in system.executionList:
                        system.executionList.append(owner)
                        system.executionGraph[owner] = set()
                        # FIX: NEED TO ALSO ADD SystemInputState (AND ??ProcessInputState) PROJECTIONS
                        # # Add self to system's list of OriginMechanisms if it doesn't have any afferents
                        # if not any(state.path_afferents for state in owner.input_states):
                        #     system.origin_mechanisms.mechs.append(owner)


