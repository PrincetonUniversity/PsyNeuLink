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

A GatingMechanism is an `AdaptiveMechanism` that modulates the value of the inputState(s) and/or outputState(s) of 
one or more `ProcessingMechanisms`.   It's function takes a value 
COMMENT:
    ??FROM WHERE?
COMMENT
and uses that to calculate a `gating_policy`:  a list of `gating <LINK>` values, one for each of states that it 
gates.  Each of these values is assigned as the value of a `GatingSignal` (a subclass of `OutputState`) in the
GatingMechanism, and used by an associated `GatingProjection` to modulate the value of the state to which it projects.  
A GatingMechanism can regulate only the parameters of mechanisms in the `System` to which it belongs. 
COMMENT: TBI
The gating components of a system can be displayed using the system's 
`show_graph` method with its **show_gating** argument assigned as :keyword:``True`.  
COMMENT
The gating components of a system are executed after all `Proces singMechanisms <ProcessingMechanism>`, 
`LearningMechanisms <LearningMechanism>`, and  `ControlMechanisms <ControlMechanism>` in that system have been executed.


.. _GatingMechanism_Creation:

Creating A GatingMechanism
---------------------------

GatingMechanisms can be created using the standard Python method of calling the constructor for the desired type.
A GatingMechanism is also created automatically if `gating is specified <GatingMechanism_Specifying_Gating>` for an 
inputState or outputState, in which case a `GatingProjection` is also automatically created that projects 
from the GatingMechanism to the specified state. How a GatingMechanism creates its `GatingProjections 
<GatingProjection>` and determines their value depends on the `subclass <GatingMechanism>`.

.. _GatingMechanism_Specifying_Gating:

Specifying gating
~~~~~~~~~~~~~~~~~

GatingMechanisms are used to modulate the value of an `inputState <InputState>` or `outputState <OutputState>`.  
An inputState or outputState can be specified for gating by assigning it a `GatingProjection` in the 
**input_states** or **output_states** arguments of the constructor for the mechanism to which it belongs 
(see `Mechanism_States <LINK>`).  The inputStates and outputStates to be gated by a GatingMechanism can also be 
specified in the  **gating_signals**  argument of the constructor for a GatingMechanism.  The **gating_signals** 
argument must be a list, each item of which must refer to a state to be gated specified in any of the following ways:

  * *InputState* or *OutputState of the Mechanism to which the state belongs;
  |
  * *tuple*, with the *name* of the state as its 1st item. and the mechanism to which it belongs as the 2nd;  
    note that this is a convenience format, which is simpler to use than a specification dictionary (see below), 
    but precludes specification of any parameters <GatingSignal_Structure>` for the GatingSignal.
  |
  * *specification dictionary*, that must contain at least the following two entries:
    * *NAME* - a string that is the name of the state to be gated;
    * *MECHANISM*:Mechanism - the Mechanism to which the state belongs. 
    The dictionary can also contain entries for any other GatingSignal parameters to be specified
    (e.g., *MODULATION_OPERATION*:ModulationOperation to specify how the value of the state will be modulated;
    see `below <GatingSignal_Structure>` for a list of parameters).

A `GatingSignal` is created for each item listed in **gating_signals**, and all of the GatingSignals for a  
GatingMechanism are listed in its `gating_signals <GatingMechanism.gating_signals>` attribute.  Each GatingSignal is 
assigned a `GatingProjection` to the inputState or outputState of the mechanism specified, that is used to modulate 
the state's value. GatingSignals are a type of `OutputState`, and so they are also listed in the GatingMechanism's 
`output_states <GatingMechanism.outut_states>` attribute.

COMMENT:
  *** PUT IN InputState AND OutputState DOCUMENTATION

  Gating can be also be specified for an `InputState` or `OutputState` when it is created in any of the following ways:

    * in a 2-item tuple, in which the first item is a `state specification <LINK>`, 
      and the second item is a `gating specification <>`

    * keywords GATE (==GATE_PRIMARY) GATE_ALL, GATE_PRIMARY
        or an entry in the state specification dictionary with the key "GATING", and a value that is the
        keyword TRUE/FALSE, ON/OFF, GATE, a ModulationOpearation value, GatingProjection, or its constructor

.. _GatingMechanism_Execution:

Execution
---------

A GatingMechanism executes in the same way as a ProcessingMechanism, based on its place in the system's 
`graph <System.graph>`.  Because GatingProjections are likely to introduce cycles (loops) in the graph,
the effects of a GatingMechanism and its projections will generally not be applied in the first
`round of execution <LINK>` (see `initialization <LINK>` for a description of how to configure the initialization
of feedback loops in a System).  When executd, a GatingMechanism uses its input to determine the value of its
`GatingSignals <GatingSignal>` and their corresponding `GatingProjections <GatingProjection>`.  In the subsequent 
round of execution , each GatingProjection's value is used by the state to which it projects to modulate the 
`value <State.value>` of that state.

When a GatingMechanism executes, the value of each item in its `gating_policy` are assigned as the values of each of
the corresponding GatingSignals in its `gating_signals` attribute.  Those, in turn, as used by their associated
`GatingProjections` to modulate the value of the state to which they project.  This is done by assigning the
GatingSignal's value to a parameter of the state's function, as specified by the GatingSignal's `modulation_operation` 
parameter (see `GatingSignal_Execution` for details). 

.. note::
   A state that receives a `GatingProjection` does not update its value until its owner mechanism executes 
   (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a GatingMechanism 
   has executed, a state that it gates will not assume its new value until the state's owner has executed.

.. _GatingMechanism_Class_Reference:

Class Reference
---------------

"""

# IMPLEMENTATION NOTE: COPIED FROM DefaultProcessingMechanism;
#                      ADD IN GENERIC CONTROL STUFF FROM DefaultGatingMechanism

from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.States.State import State_Base, _instantiate_state, _parse_state_spec
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingSignal import GatingSignal
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingSignal import gating_signal_keywords
from PsyNeuLink.Components.Projections.Projection import _validate_projection_receiver_mech

GatingMechanismRegistry = {}


class GatingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class GatingMechanism(AdaptiveMechanism_Base):
    """
    GatingMechanism_Base(     \
    default_input_value=None, \
    gating_signals=None       \
    function=Linear,          \
    params=None,              \
    name=None,                \
    prefs=None)

    Abstract class for GatingMechanism.

    .. note::
       GatingMechanisms should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the constructor for a :doc:`subclass <GatingMechanism>`.

    COMMENT:
        Description:
            # VERIFY:
            Protocol for instantiating unassigned GatingProjections (i.e., w/o a sender specified):
               If sender is not specified for a GatingProjection (e.g., in an inputState or OutputState tuple spec) 
                   it is flagged for deferred_init() in its __init__ method
               When the next GatingMechanism is instantiated, if its params[MAKE_DEFAULT_GATING_MECHANISM] == True, its
                   _take_over_as_default_gating_mechanism method is called in _instantiate_attributes_after_function;
                   it then iterates through all of the inputStates and outputStates of all of the mechanisms in its 
                   system, identifies ones without a sender specified, calls its deferred_init() method,
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

    gating_signals : List[InputState or OutputState, tuple[str, Mechanism], or dict]
        specifies the inputStates and/or outputStates to be gated by the GatingMechanism;
        the number of items must equal the length of the **default_gating_policy** arg 
        (see `gating_signals <GatingMechanism.gating_signals>` for details).

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored output states.
        
    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters
        for the mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default ControlMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for the mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    gating_signals : List[GatingSignal]
        list of `GatingSignals <ControlSignals>` for the GatingMechanism, each of which sends a `GatingProjection`
        to the `inputState <InputState>` or `outputState <OutputState>` that it gates (same as GatingMechanism's 
        `output_states <Mechanism.output_states>` attribute).

    gating_projections : List[GatingProjection]
        list of `GatingProjections <GatingProjection>`, one for each `GatingSignal` in `gating_signals`.

    gating_policy : 2d np.array
        each items is the value assigned to the corresponding GatingSignal listed in `gating_signals`
        (same as the GatingMechanism's `value <Mechanism.value>` attribute).
        
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
                 function = Linear(slope=1, intercept=0),
                 gating_signals:tc.optional(list) = None,
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

        if self.gating_signals:
            for gating_signal in self.gating_signals:
                self._instantiate_gating_signal(gating_signal=gating_signal, context=context)

        # IMPLEMENTATION NOTE:  Don't want to call this because it instantiates undesired default outputState
        # super()._instantiate_output_states(context=context)

    def _instantiate_attributes_after_function(self, context=None):
        """Take over as default GatingMechanism (if specified) and implement any specified GatingProjections
        """
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
        # Check the input_states and output_states of the system's mechanisms
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


    def _instantiate_gating_signal(self, gating_signal=None, context=None):
        """Instantiate OutputState for GatingSignal and assign (if specified) or instantiate GatingProjection

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
             so that correct outputState can be assigned as its sender;
        * index of outputState is incremented based on number of GatingSignals already instantiated;

        Returns GatingSignal (OutputState)
        """
        from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

        # EXTEND gating_policy TO ACCOMMODATE NEW GatingSignal -------------------------------------------------
        #        also used to determine constraint on GatingSignal value

        if self.gating_policy is None:
            self.gating_policy = np.array(defaultControlAllocation)
        else:
            self.gating_policy = np.append(self.gating_policy, defaultGatingPolicy)

        # PARSE gating_signal SPECIFICATION -----------------------------------------------------------------------

        gating_projection = None
        gating_signal_params = None

        gating_signal_spec = _parse_gating_signal_spec(owner=self, state_spec=gating_signal)

        # Specification is a GatingSignal (either passed in directly, or parsed from tuple above)
        if GATING_SIGNAL in gating_signal_spec:
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

            # IMPLEMENTATION NOTE:
            #    THIS IS TO HANDLE FUTURE POSSIBILITY OF MULTIPLE GatingProjections FROM A SINGLE GatingSignal;
            #    FOR NOW, HOWEVER, ONLY A SINGLE ONE IS SUPPORTED
            # parameter_states = [proj.recvr for proj in control_projections]
            if len(gating_projections) > 1:
                raise GatingMechanismError("PROGRAM ERROR: list of ControlProjections is not currently supported "
                                            "as specification in a ControlSignal")
            else:
                gating_projection = gating_projections[0]
                state = gating_projection.receiver

        # Instantiate OutputState for GatingSignal
        else:

            gating_signal_name = gating_signal_spec[NAME]
            # FIX: CALL REGISTRY FOR NAME HERE (AS FOR OUTPUTSTATE IN MECHANISM
            if self.gating_signals and gating_signal_name in self.gating_signals.names:
                gating_signal_name = gating_signal_name + '-' + repr(len(self.gating_signals))

            # from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingSignal import GatingSignal
            # from PsyNeuLink.Components.States.State import _instantiate_state

            # Get constraint for OutputState's value
            #    - get GatingMechanism's value
            self._update_value(context=context)
            # - get OutputState's index
            try:
                output_state_index = len(self.output_states)
            except (AttributeError, TypeError):
                output_state_index = 0
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

        # Validate gating_projection (if specified) and get receiver's name
        if gating_projection:
            _validate_projection_receiver_mech(self, gating_projection, context=context)

            if not isinstance(gating_projection, GatingProjection):
                raise GatingMechanismError("PROGRAM ERROR: Attempt to assign {}, "
                                                  "that is not a GatingProjection, to GatingSignal of {}".
                                                  format(gating_projection, self.name))
            if gating_projection.value is DEFERRED_INITIALIZATION:
                gating_projection.init_args['sender']=gating_signal
                if gating_projection.init_args['name'] is None:
                    # FIX 5/23/17: CLEAN UP NAME STUFF BELOW:
                    gating_projection.init_args['name'] = CONTROL_PROJECTION + \
                                                   ' for ' + state.owner.name + ' ' + state.name
                gating_projection._deferred_init()
            else:
                gating_projection.sender = gating_signal

        # Instantiate GatingProjection
        else:
            # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
            gating_projection = GatingProjection(sender=gating_signal,
                                           receiver=state,
                                           name=GATING_PROJECTION + gating_signal_name)

        # Add GatingProjection to list of outputState's outgoing projections
        # (note: if it was deferred, it just added itself, skip)
        if not gating_projection in gating_signal.efferents:
            gating_signal.efferents.append(gating_projection)

        # Add GatingProjection to GatingMechanism's list of GatingProjections
        try:
            self.gating_projections.append(gating_projection)
        except AttributeError:
            self.gating_projections = [gating_projection]

        # UPDATE output_states AND gating_projections -------------------------------------------------------------

        try:
            self.output_states[gating_signal.name] = gating_signal
        except (AttributeError, TypeError):
            self.output_states = ContentAddressableList(component_type=State_Base, list=[gating_signal])

        # Add index assignment to outputState
        gating_signal.index = output_state_index

        # (Re-)assign control_signals attribute to output_states
        self.gating_signals = self.output_states

        return gating_signal

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Updates GatingProjections based on inputs

        Must be overriden by subclass
        """
        # FIX: THIS NEEDS TO BE IMPLEMENTED
        return [0]
        # raise GatingMechanismError("{0} must implement execute() method".format(self.__class__.__name__))

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


def _parse_gating_signal_spec(owner, state_spec):
    """Take specifications for one or more states to be gated, and return GatingSignal specification dictionary

    state_spec can take any of the following forms:
        - an existing GatingSignal
        - an existing InputState or OutputState for a Mechanisms in self.system
        - a list of state specifications (see below)
        - a dictionary that contains either a:
            - single state specification:
                NAME:str - contains the name of an InputState or OutputState belonging to MECHANISM
                MECHANISM:Mechanism - contains a reference to a Mechanism in self.system that owns NAME'd state
                <PARAM_KEYWORD>:<GatingSignal param value>
            - multiple state specification:
                NAME:str - used as name of GatingSignal
                STATES:List[tuple, dict] - each item must be state specification tuple or dict
                <PARAM_KEYWORD>:<GatingSignal param value>
    
    Each state specification must be a:
        - (str, Mechanism) tuple
        - {NAME:str, MECHANISM:Mechanism} dict
        where:
            str is the name of an InputState or OutputState of the Mechanism,
            Mechanism is a reference to an existing that belongs to self.system 
    
    Checks for duplicate state specifications within state_spec or with any existing GatingSignal of the owner
        (i.e., states that will receive more than one GatingProjection from the owner)
        
    If state_spec is already a GatingSignal, it is returned (in the GATING_SIGNAL entry) along with its parsed elements 
    
    Returns dictionary with the following entries:
        NAME:str - name of either the gated state (if there is only one) or the GatingSignal
        STATES:list - list of states to be gated
        GATING_SIGNAL:GatingSignal or None
        PARAMS:dict - params dict if any were included in the state_spec
    """
    
    from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingSignal import GatingSignal
    from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

    GATING_SIGNAL_SUFFIX = '_' + GatingSignal.__name__
    DEFAULT_GATING_SIGNAL_NAME = 'Default'+ GATING_SIGNAL_SUFFIX

    gating_signal = None
    states = []
    params = {}
    state_name = None
    mech = None

    # def _get_default_gating_signal_name():
    #     """Return default name for gating signal
    #     Default_GatingSignal if it is the first,
    #     Default_GatingSignal-2 for second
    #     Default_GatingSignal-3 for third, etc.
    #     """
    #
    #     # FIX: USE REGISTRY HERE (AS FOR OUTPUTSTATE REGISTRY ON MECHANISM)
    #     if owner.gating_signals:
    #         # # Get the number of existing gating_signals with the default name
    #         # index = len([gs for gs in owner.gating_signals if ('Default'+ GATING_SIGNAL_SUFFIX) in gs.name])
    #     else:
    #         index = ''
    #     if index:
    #         index = repr(index+1)
    #     return 'Default'+GATING_SIGNAL_SUFFIX+index

    # Specification is for a GatingSignal - return as is
    if isinstance(state_spec, GatingSignal):
        #  Check that any GatingProjections it has are to mechanisms in self.system
        if not all(gating_proj.receiver.owner in owner.system.mechanisms
                   for gating_proj in state_spec.efferents):
            raise GatingMechanismError("The GatingSignal specified in the {} arg for {} ({}) "
                                        "has one more more GatingProjections to a mechanism "
                                        "that is not in {}".
                                        format(GATING_SIGNALS, owner.name, state_spec.name, owner.system.name))
        # return state_spec
        gating_signal = state_spec
        gating_signal_name = gating_signal.name
        states = [proj.receiver.owner for proj in gating_signal.efferents]
        if not states:
            raise GatingMechanismError("Attempt to assign an existing {} to {} that has no GatingProjections".
                                       format(GATING_SIGNAL, owner.name))

    # For all other specs:
    #    - if it is a single spec (state name and mech):
    #        validate that the mech is in self.system, has a state of that name, and then return the state
    #    - if it is a list:
    #        iterate through list, calling _parse_gating_signal_spec recursively, to build up the list of states

    # Specification is for an existing GatingProjection
    #    so check if it is to a state of a mechanism in self.system
    elif isinstance(state_spec, GatingProjection):
        if not state_spec.receiver.owner in owner.system.mechanisms:
            raise GatingMechanismError("The GatingSignal specified in the {} arg for {} ({}) "
                                       "has one more more GatingProjections to a mechanism "
                                       "that is not in {}".
                                       format(GATING_SIGNALS, owner.name, state_spec.name, owner.system.name))
        state_name = state_spec.receiver.name
        gating_signal_name = state_name + GATING_SIGNAL_SUFFIX
        mech = state_spec.reciever.owner

    # Specification is for an existing InputState or OutputState,
    #    so check that it's owner belongs to self.system
    elif isinstance(state_spec, (InputState, OutputState)):
        if not state_spec.owner.system in owner.system.mechanisms:
            raise GatingMechanismError("The state specified in the {} arg for {} ({}) "
                                        "belongs to a mechanism that is not in the same system ({})".
                                        format(GATING_SIGNALS, owner.name,
                                               state_spec.name,
                                               state_spec.owner.system.name))
        state_name = state_spec.name
        gating_signal_name = state_name + GATING_SIGNAL_SUFFIX
        mech = state_spec.owner

    elif isinstance(state_spec, tuple):
        state_name = state_spec[0]
        mech = state_spec[1]
        gating_signal_name = state_name + GATING_SIGNAL_SUFFIX
        # Check that 1st item is a str (presumably the name of one of the mechanism's states)
        if not isinstance(state_name, str):
            raise GatingMechanismError("1st item of specification tuple for the state to be gated by {} of {} ({})"
                                       "must be a string that is the name of the state".
                                       format(GATING_SIGNAL, owner.name, state_name))
        # Check that 2nd item is a mechanism
        if not isinstance(mech, Mechanism):
            raise GatingMechanismError("2nd item of specification tuple for the state to be gated by {} of {} ({})"
                                       "must be a Mechanism that is the mech to which the state {} belongs".
                                       format(GATING_SIGNAL, owner.name, mech, state_name))

    # Specification is a list, presumably of one or more states specs
    elif isinstance(state_spec, list):
        # Validate each item in the list (which should be a state state_spec), and
        #    - add the state(s) returned to state list
        #    - assign state_name as None,
        #        since there is no single name that can be used as the name for the GatingSignal
        gating_signal_name = DEFAULT_GATING_SIGNAL_NAME
        for spec in state_spec:
            spec_dict = _parse_gating_signal_spec(owner, spec)
            states.extend(spec_dict[STATES])

    # Specification is a dict that could be for a single state state_spec or a list of ones
    elif isinstance(state_spec, dict):

        # If it has a STATES entry, it must be for a list
        if STATES in state_spec:
            # Validate that the STATES entry has a list
            state_specs = state_spec[STATES]
            if not isinstance(state_specs, list):
                raise GatingMechanismError("The {} entry of the dict in the {} arg for {} must be "
                                           "a list of state specifications".
                                            format(STATES, GATING_SIGNALS, owner.name))
            # Validate each item in the list (which should be a state state_spec), and
            #    - add the state(s) returned to state list
            #    - assign state_name to the NAME entry
            #        (which will be used as the name for the GatingSignal in _instantiate_gating_signal)
            for spec in state_specs:
                spec_dict = _parse_gating_signal_spec(owner, spec)
                states.extend(spec_dict[STATES])
            if NAME in state_spec:
                state_name = state_spec[NAME]
                gating_signal_name = state_name
            else:
                gating_signal_name = DEFAULT_GATING_SIGNAL_NAME

        # If it doesn't have a STATES entry
        else:
            # If there is a non-keyword key, treat as the name to be used for the GatingSignal,
            #    and the value a state spec or list of ones
            state_name = next((key for key in state_spec if not key in gating_signal_keywords), None)
            if state_name:
                gating_signal_name = state_name
                spec_dict = _parse_gating_signal_spec(owner, state_spec[gating_signal_name])
                states = spec_dict[STATES]
            # Otherwise, it must be for a single state state_spec,
            #    which means it must have a NAME and a MECHANISM entry:
            else:
                if not NAME in state_spec:
                    raise GatingMechanismError("Specification dict for the state to be gated by {} of {} must have a "
                                               "NAME entry that is the name of the state".
                                               format(GATING_SIGNAL, owner.name))
                state_name = state_spec[NAME]
                gating_signal_name = state_name + GATING_SIGNAL_SUFFIX

                # GatingSignal projects to a single state (named in NAME entry)
                if not MECHANISM in state_spec:
                    raise GatingMechanismError("Specification dict for state to be gated by {} of {} ({}) must have a "
                                               "MECHANISM entry specifying the mechanism to which the state belongs".
                                               format(GATING_SIGNAL, owner.name, state_name))
                mech = state_spec[MECHANISM]

        # Check that all of the other entries in the dict are for valid GatingSignal params
        #    - skip any entries specifying gating signal
        #    - place others in params
        for param_entry in [entry for entry in state_spec if not entry in {gating_signal_name, MECHANISM}]:
            if not param_entry in gating_signal_keywords:
                raise GatingMechanismError("Entry in specification dictionary for {} arg of {} ({}) "
                                           "is not a valid {} parameter".
                                           format(GATING_SIGNAL, owner.name, param_entry,
                                                  GatingSignal.__name__))
            params[param_entry] = state_spec[param_entry]

    else:
        # raise GatingMechanismError("PROGRAM ERROR: unrecognized GatingSignal specification for {} ({})".
        #                             format(self.name, state_spec))
        raise GatingMechanismError("Specification of {} for {} ({}) must be an InputState or OutputState, "
                                   "a tuple specifying a name for one and a mechanism to which it belongs ,"
                                   "a list of state specifications, "
                                   "a {} specification dict with one or more state specifications and "
                                   "entries for {} parameters, or an existing GatingSignal".
                                    format(GATING_SIGNAL, owner.name, state_spec, GATING_SIGNAL, GATING_SIGNAL))

    # If a states list has not already been constructed, do so here
    if not states:
        # Check that specified state is an InputState or OutputState of the Mechanism
        if state_name in mech.input_states:
            state_type = INPUT_STATE
            state = mech.input_states[state_name]
        elif state_name in mech.output_states:
            state_type = OUTPUT_STATE
            state = mech.output_states[state_name]
        else:
            raise GatingMechanismError("{} (in specification of {}  {}) is not an "
                                       "InputState or OutputState of {}".
                                        format(state_name, GATING_SIGNAL, owner.name, mech))
        # Check that the Mechanism is in GatingMechanism's system
        if owner.system and not mech in owner.system.mechanisms:
            raise GatingMechanismError("Specification in {} arg for {} ({} {} of {}) "
                                        "must be for a Mechanism in {}".
                                        format(GATING_SIGNALS,
                                               owner.name,
                                               state_name,
                                               state_type,
                                               mech.name,
                                               owner.system.name))
        states = [state]

    # Check for any duplicate states in specification for this GatingSignal or existing ones for the owner
    all_states = []
    # Get states gated in any already instantiated GatingSignals
    if owner.gating_signals:
        for gs in owner.gating_signals:
            all_states.extend([proj.receiver.owner for proj in gs.efferents])
    # Add states for current GatingSignal
    all_states.extend(states)
    # Check for duplicates
    # if any(thelist.count(x) > 1 for x in thelist)
    if len(all_states) != len(set(all_states)):
        for test_state in all_states:
            if next((test_state == state  for state in all_states), None):
                raise GatingMechanismError("{} receives more than one GatingProjection from the {}s in {}".
                                           format(test_state, GatingSignal.__name__, owner.name))
        raise GatingMechanismError("PROGRAM ERROR: duplicate state detected in {} specifications for {} ({})"
                                   "but could not find the offending state".
                                   format(GATING_SIGNAL, owner.name, gating_signal_name))

    return {NAME: gating_signal_name,
            STATES: states,
            PARAMS: params,
            GATING_SIGNAL: gating_signal}