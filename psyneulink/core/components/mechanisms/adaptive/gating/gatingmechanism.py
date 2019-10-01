# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  GatingMechanism ************************************************

"""
Sections
--------

  * `GatingMechanism_Overview`
  * `GatingMechanism_Creation`
  * `GatingMechanism_Structure`
      - `GatingMechanism_Input`
      - `GatingMechanism_Function`
  * `GatingMechanism_Execution`
  * `GatingMechanism_Class_Reference`

.. _GatingMechanism_Overview:

Overview
--------

A GatingMechanism is a subclass of `ModulatoryMechanism` that is restricted to using only `GatingSignals
<GatingSignal>` and not ControlSignals.  Accordingly, its constructor has a **gating_signals** argument in place of
a **modulatory_signals** argument.  It also lacks any attributes related to control, including this related to cost
and net_outcome.  In all other respects it is identical to its parent class, ModulatoryMechanism.

.. _GatingMechanism_Creation:

Creating A GatingMechanism
---------------------------

A GatingMechanism is created by calling its constructor.  When a GatingMechanism is created, the OutputStates it
monitors and the `InputStates <InputState>` and/or `OutputStates <OutputState>` it modulates can be specified in the 
**montior_for_modulation** and **objective_mechanism** arguments of its constructor, respectively.  Each can be 
specified in several ways, as described in `ModulatoryMechanism_Monitor_for_Modulation` and 
`ModulatoryMechanism_Modulatory_Signals` respectively. If neither of those arguments is specified, then only the 
GatingMechanism is constructed, and its inputs and the InputStates and/or OutputStates it modulates must be specified 
in some other way.
COMMENT:
TBI FOR COMPOSITION
A GatingMechanism is also created automatically if `gating
is specified <GatingMechanism_Specifying_Gating>` for an `InputState`, `OutputState` or `Mechanism <Mechanism>`,
in which case a `GatingProjection` is automatically created that projects from the GatingMechanism to the specified
target.
COMMENT

.. _GatingMechanism_Specifying_Gating:

*Specifying gating*
~~~~~~~~~~~~~~~~~~~

A GatingMechanism is used to modulate the value of an `InputState` or `OutputState`. An InputState or OutputState can
be specified for gating by assigning it a `GatingProjection` or `GatingSignal` anywhere that the Projections to a State
or its `ModulatorySignals can be specified <State_Creation>`.  A `Mechanism <Mechanism>` can also be specified for
gating, in which case the `primary InputState <InputState_Primary>` of the specified Mechanism is used.  States
(and/or Mechanisms) can also be specified in the  **gating_signals** argument of the constructor for a
GatingMechanism. The **gating_signals** argument must be a list, each item of which must refer to one or more States
(or the Mechanism(s) to which they belong) to be gated by that GatingSignal.  The specification for each item in the
list can use any of the forms used to `specify a GatingSignal <GatingSignal_Specification>`.

.. _GatingMechanism_GatingSignals:

GatingSignals
^^^^^^^^^^^^^

A `GatingSignal` is created for each item listed in the **gating_signals** argument of the constructor, and all of the
GatingSignals for a GatingMechanism are listed in its `gating_signals <GatingMechanism.gating_signals>` attribute.
Each GatingSignal is assigned one or more `GatingProjections <GatingProjection>` to the InputState(s) and/or
OutputState(s) it gates. By default, the `function <GatingMechanism.function>` of GatingMechanism generates a `value
<GatingMechanism.value>` -- its `gating_allocation <GatingSignal.gating_allocation>` -- with a single item, that is
used by all of the GatingMechanism's GatingSignals.  However,  if a custom `function <GatingMechanism.function>` is
specified that generates a `gating_allocation <GatingSignal.gating_allocation>` with more than one item, different
GatingSignals can be assigned to the different items (see `GatingMechanism_Function` below).

.. _GatingMechanism_Modulation:

Modulation
^^^^^^^^^^

Each GatingMechanism has a `modulation <GatingSignal.modulation>` attribute, that provides a default for the way
in which its GatingSignals modulate the value of the States they gate (see `modulation <ModulatorySignal_Modulation>` 
for an explanation of how this attribute is specified and used to modulate the value of a State).  Each GatingSignal 
uses this value, unless its value is `individually specified <GatingSignal_Modulation>`.

.. _GatingMechanism_Structure:

Structure
---------

.. _GatingMechanism_Input:

*Input*
~~~~~~~

The input to a GatingMechanism is determined in the same manner as the `input <ModulatoryMechanism_Input>` to
any `ModulatoryMechanism`.

.. _GatingMechanism_Function:

*Function*
~~~~~~~~~~

A GatingMechanism's `function <GatingMechanism.function>` is determined and operates in the same manner as the
`function <ModulatoryMechanism_Function>` of any `ModulatoryMechanism`.

.. _GatingMechanism_Output:

*Output*
~~~~~~~~

The OutputStates of a GatingMechanism are `GatingSignals <GatingSignal>` (listed in its `gating_signals
<GatingMechanism.gating_signals>` attribute). It  has a `GatingSignal` for each `InputState` and/or `OutputState` 
specified in the **gating_signals** argument of its constructor, that sends a `GatingProjection` to those States.  
The GatingSignals are listed in the `gating_signals <GatingMechanism.gating_signals>` attribute;  since they are a 
type of `OutputState`, they are also listed in the GatingMechanism's `output_states <GatingMechanism.output_states>` 
attribute. The InputStates and/or OutputStates modulated by a GatingMechanism's GatingSignals can be displayed using 
its `show <GatingMechanism.show>` method. If the GatingMechanism's `function <GatingMechanism.function>` generates a
`gating_allocation <GatingMechanism.gating_allocation>` with a single value (the default), then this is used as the
`allocation <GatingSignal.alloction>` for all of the GatingMechanism's `gating_signals
<GatingMechanism.gating_signals>`.  If the `gating_allocation <GatingMechanism.gating_allocation>` has multiple
items, and this is the same as the number of GatingSignals, then each GatingSignal is assigned the value of the
corresponding item in the `gating_allocation <GatingMechanism.gating_allocation>`.  If there is a different number of
`gating_signals <GatingMechanism.gating_signals>` than the number of items in the `gating_allocation
<GatingMechanism.gating_allocation>`, then the `index <GatingSignal.index>` attribute of each GatingSignal must be
specified (e.g., in a `specification dictionary <GatingSignal_Specification>` in the **gating_signal** argument of
the GatingMechanism's constructor), or an error is generated.  The `default_allocation
<GatingMechanism.default_allocation>` attribute can be used to specify a  default allocation for GatingSignals that
have not been assigned their own `default_allocation  <GatingSignal.default_allocation>`. The `allocation
<GatingSignal.allocation>` is used by each GatingSignal to determine its `intensity  <GatingSignal.intensity>`,
which is then assigned to the `value <GatingProjection.value>` of the GatingSignal's `GatingProjection`.   The `value
<GatingProjection.value>` of the GatingProjection is used to modify the value of the InputState and/or OutputState it
gates (see `GatingSignal_Modulation` for description of how a GatingSignal modulates the value of a parameter).

.. _GatingMechanism_Execution:

Execution
---------

A GatingMechanism executes in the same way as a `ProcessingMechanism <ProcessingMechanism>`, based on its place in the
Composition's `graph <Composition.graph>`.  Because `GatingProjections <GatingProjection>` are likely to introduce
cycles (recurrent connection loops) in the graph, the effects of a GatingMechanism and its projections will generally
not be applied in the first `TRIAL` (see
COMMENT:
`Composition_Initial_Values_and_Feedback` and
COMMENT
**feedback** argument for the `add_projection <Composition.add_projection>`
method of `Composition` for a description of how to configure the initialization of feedback loops in a Composition;
also see `Scheduler` for a description of detailed ways in which a GatingMechanism and its dependents can be scheduled
to execute).

When executed, a GatingMechanism  uses its input to determine the value of its `gating_allocation
<GatingMechanism.gating_allocation>`, each item of which is used by a corresponding `GatingSignal` to determine its
`gating_signal <GatingSignal.gating_signal>` and assign to its `GatingProjections <GatingProjection>`. In the
subsequent `TRIAL`, each GatingProjection's value is used by the State to which it projects to modulate the `value
<State_Base.value>` of that State (see `modulation <ModulatorySignal_Modulation>` fon an explanation of how the value
of a State is modulated).

.. note::
   A State that receives a `GatingProjection` does not update its `value <State_Base.value>` (and therefore does not
   reflect the influence of its `GatingSignal`) until that State's owner Mechanism executes
   (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

.. _GatingMechanism_Class_Reference:

Class Reference
---------------

"""

import warnings

import numpy as np
import typecheck as tc

from psyneulink.core.components.mechanisms.adaptive.modulatorymechanism import ModulatoryMechanism
from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultGatingAllocation
from psyneulink.core.globals.keywords import \
    GATING, GATING_PROJECTION, GATING_PROJECTIONS,GATING_SIGNAL,GATING_SIGNALS,GATING_SIGNAL_SPECS, \
    INIT_EXECUTE_METHOD_ONLY, MAKE_DEFAULT_GATING_MECHANISM, MULTIPLICATIVE, PROJECTION_TYPE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import ContentAddressableList

__all__ = [
    'GatingMechanism', 'GatingMechanismError', 'GatingMechanismRegistry'
]

GatingMechanismRegistry = {}


def _is_gating_spec(spec):
    from psyneulink.core.components.projections.modulatory.gatingprojection import GatingProjection
    if isinstance(spec, tuple):
        return any(_is_gating_spec(item) for item in spec)
    if isinstance(spec, dict) and PROJECTION_TYPE in spec:
        return _is_gating_spec(spec[PROJECTION_TYPE])
    elif isinstance(spec, (GatingMechanism,
                           GatingSignal,
                           GatingProjection,
                           ModulatoryMechanism)):
        return True
    elif isinstance(spec, type) and issubclass(spec, (GatingSignal,
                                                      GatingProjection,
                                                      GatingMechanism,
                                                      ModulatoryMechanism)):
        return True
    elif isinstance(spec, str) and spec in {GATING, GATING_PROJECTION, GATING_SIGNAL}:
        return True
    else:
        return False


class GatingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

def _gating_allocation_getter(owning_component=None, context=None):
    return owning_component.modulatory_allocation

def _gating_allocation_setter(value, owning_component=None, context=None):
    owning_component.parameters.modulatory_allocation._set(np.array(value), context)
    return value

def _control_allocation_getter(owning_component=None, context=None):
    from psyneulink.core.components.mechanisms.adaptive.control import ControlMechanism
    from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
    raise GatingMechanismError(f"'control_allocation' attribute is not implemented on {owning_component.name};  "
                                f"consider using a {ControlMechanism.__name__} instead, "
                                f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                f"{GatingSignal.__name__}s are needed.")

def _control_allocation_setter(value, owning_component=None, context=None, **kwargs):
    from psyneulink.core.components.mechanisms.adaptive.control import ControlMechanism
    from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
    raise GatingMechanismError(f"'control_allocation' attribute is not implemented on {owning_component.name};  "
                                f"consider using a {ControlMechanism.__name__} instead, "
                                f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                f"{GatingSignal.__name__}s are needed.")


class GatingMechanism(ModulatoryMechanism):
    """
    GatingMechanism(                                \
        default_gating_allocation=None,             \
        size=None,                                  \
        function=Linear(slope=1, intercept=0),      \
        default_allocation=None,                    \
        gating_signals:tc.optional(list) = None,    \
        modulation=MULTIPLICATIVE,                  \
        params=None,                                \
        name=None,                                  \
        prefs=None)

    Subclass of `AdaptiveMechanism <AdaptiveMechanism>` that gates (modulates) the value(s)
    of one or more `States <State>`.

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

    default_gating_allocation : value, list or ndarray : default `defaultGatingAllocation`
        the default value for each of the GatingMechanism's GatingSignals;
        its length must equal the number of items specified in the **gating_signals** argument.

    size : int, list or 1d np.array of ints
        specifies default_gating_allocation as an array of zeros if **default_gating_allocation** is not passed as an
        argument;  if **default_gating_allocation** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies the function used to transform the GatingMechanism's `variable <GatingMechanism.variable>`
        to a `gating_allocation`.

    default_allocation : number, list or 1d array : None
        specifies the default_allocation of any `gating_signals <GatingMechanism.gating.signals>` for
        which the **default_allocation** was not specified in its constructor (see default_allocation
        <GatingMechanism.default_allocation>` for additional details).

    gating_signals : List[GatingSignal, InputState, OutputState, Mechanism, tuple[str, Mechanism], or dict]
        specifies the `InputStates <InputState>` and/or `OutputStates <OutputStates>` to be gated by the
        GatingMechanism; the number of items must equal the length of the **default_gating_allocation**
        argument; if a `Mechanism <Mechanism>` is specified, its `primary InputState <InputState_Primary>`
        is used (see `GatingMechanism_GatingSignals for details).

    modulation : ModulationParam : MULTIPLICATIVE
        specifies the default form of modulation used by the GatingMechanism's `GatingSignals <GatingSignal>`,
        unless they are `individually specified <GatingSignal_Specification>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <GatingMechanism.name>`
        specifies the name of the GatingMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the GatingMechanism; see `prefs <GatingMechanism.prefs>` for details.


    Attributes
    ----------

    variable : value, list or ndarray
        used as the input to the GatingMechanism's `function <GatingMechanism.function>`.  Its format is determined
        by the **default_gating_allocation** or **size** argument of the GatingMechanism's constructor (see above),
        and is the same format as its `gating_allocation <GatingMechanis.gating_allocation>` (unless a custom
        `function <GatingMechanism.function>` has been assigned).

    function : TransferFunction
        determines the function used to transform the GatingMechanism's `variable <GatingMechanism.variable>`
        to a `gating_allocation`;  the default is an identity function that simply assigns
        `variable <GatingMechanism.variable>` as the `gating_allocation <GatingMechanism.gating_allocation>`.

    default_allocation : number, list or 1d array
        determines the default_allocation of any `gating_signals <GatingMechanism.gating.signals>` for
        which the **default_allocation** was not specified in its constructor;  if it is None (not specified)
        then the GatingSignal's parameters.allocation.default_value is used. See documentation for
        **default_allocation** argument of GatingSignal constructor for additional details.

    gating_allocation : 2d array
        each item is the value assigned as the `allocation <GatingSignal.allocation>` for the corresponding
        GatingSignal listed in the `gating_signals` attribute;  the gating_allocation is the same as the
        GatingMechanism's `value <Mechanism_Base.value>` attribute).

    gating_signals : ContentAddressableList[GatingSignal]
        list of `GatingSignals <GatingSignals>` for the GatingMechanism, each of which sends
        `GatingProjection(s) <GatingProjection>` to the `InputState(s) <InputState>` and/or `OutputStates <OutputState>`
        that it gates; same as GatingMechanism `output_states <Mechanism_Base.output_states>` attribute.

    gating_projections : List[GatingProjection]
        list of all of the `GatingProjections <GatingProjection>` assigned to the GatingMechanism's
        `GatingSignals <GatingSignal>` (i.e., listed in its `gating_signals <GatingMechanism.gating_signals>` attribute.

    value : scalar or 1d np.array of ints
        the result of the GatingMechanism's `function <GatingProjection.funtion>`;
        each item is the value assigned to the corresponding GatingSignal listed in `gating_signals`,
        and used by each GatingSignal to generate the `gating_signal <GatingSignal.gating_signal>` assigned to its
        `GatingProjections <GatingProjection>`;
        same as the GatingMechanism's `gating_allocation <GatingMechanism.gating_allocation>` attribute.
        Default is a single item used by all of the `gating_signals`.

    gating_allocation : scalar or 1d np.array of ints
        the result of the GatingMechanism's `function <GatingProjection.function>`;
        each item is the value assigned to the corresponding GatingSignal listed in `gating_signals`,
        and used by each GatingSignal to generate the `gating_signal <GatingSignal.gating_signal>` assigned to its
        `GatingProjections <GatingProjection>`; same as the GatingMechanism's `value <GatingMechanism.value>` attribute.
        Default is a single item used by all of the `gating_signals`.


    modulation : ModulationParam
        the default form of modulation used by the GatingMechanism's `GatingSignals <GatingSignal>`,
        unless they are `individually specified <GatingSignal_Specification>`.

    name : str
        the name of the GatingMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the GatingMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentType = "GatingMechanism"

    initMethod = INIT_EXECUTE_METHOD_ONLY

    outputStateTypes = GatingSignal
    stateListAttr = ModulatoryMechanism.stateListAttr.copy()
    stateListAttr.update({GatingSignal:GATING_SIGNALS})


    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'GatingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # Override gating_allocatdion and suppress control_allocation
    class Parameters(ModulatoryMechanism.Parameters):
        """
            Attributes
            ----------

                value
                    see `value <GatingMechanism.value>`

                    :default value: numpy.array([0.5])
                    :type: numpy.ndarray

                control_allocation
                    see `control_allocation <GatingMechanism.control_allocation>`

                    :default value: NotImplemented
                    :type: <class 'NotImplementedType'>
                    :read only: True

                gating_allocation
                    see `gating_allocation <GatingMechanism.gating_allocation>`

                    :default value: numpy.array([0.5])
                    :type: numpy.ndarray
                    :read only: True

        """
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        value = Parameter(np.array([defaultGatingAllocation]), aliases='modulatory_allocation')
        gating_allocation = Parameter(np.array([defaultGatingAllocation]),
                                      getter=_gating_allocation_getter,
                                      setter=_gating_allocation_setter,
                                      read_only=True)
        control_allocation = Parameter(NotImplemented,
                                      getter=_control_allocation_getter,
                                      setter=_control_allocation_setter,
                                      read_only=True)

    @tc.typecheck
    def __init__(self,
                 default_gating_allocation=None,
                 size=None,
                 function=None,
                 default_allocation:tc.optional(tc.any(int, float, list, np.ndarray))=None,
                 gating_signals:tc.optional(list) = None,
                 modulation:tc.optional(str)=MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        super().__init__(default_variable=default_gating_allocation,
                         size=size,
                         function=function,
                         default_allocation=default_allocation,
                         modulatory_signals=gating_signals,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,

                         **kwargs)

    def _instantiate_output_states(self, context=None):
        self._register_modulatory_signal_type(GatingSignal,context)
        super()._instantiate_output_states(context)

    def _instantiate_gating_signal(self, gating_signal, context=None):
        """Instantiate GatingSignal OutputState and assign (if specified) or instantiate GatingProjection
        Return GatingSignal (OutputState)
        """
        return super()._instantiate_modulatory_signal(modulatory_signal=gating_signal, context=context)

    def _instantiate_attributes_after_function(self, context=None):
        """Take over as default GatingMechanism (if specified) and implement any specified GatingProjections
        """

        super()._instantiate_attributes_after_function(context=context)

        if MAKE_DEFAULT_GATING_MECHANISM in self.paramsCurrent:
            if self.paramsCurrent[MAKE_DEFAULT_GATING_MECHANISM]:
                self._assign_as_gating_mechanism(context=context)

        # FIX: 5/23/17 CONSOLIDATE/SIMPLIFY THIS RE: gating_signal ARG??  USE OF PROJECTIONS, ETC.
        # FIX:         ?? WHERE WOULD GATING_PROJECTIONS HAVE BEEN SPECIFIED IN paramsCURRENT??
        # FIX:         DOCUMENT THAT VALUE OF GATING ENTRY CAN BE A PROJECTION
        # FIX:         RE-WRITE parse_state_spec TO TAKE TUPLE THAT SPECIFIES (PARAM VALUE, GATING SIGNAL)
        #                       RATHER THAN (PARAM VALUE, GATING PROJECTION)
        # FIX: NOT CLEAR THIS IS GETTING USED AT ALL; ALSO, ??REDUNDANT WITH CALL IN _instantiate_output_states
        # If GatingProjections were specified, implement them
        if GATING_PROJECTIONS in self.paramsCurrent:
            if self.paramsCurrent[GATING_PROJECTIONS]:
                for key, projection in self.paramsCurrent[GATING_PROJECTIONS].items():
                    self._instantiate_gating_projection(projection, context=context)

    def _assign_as_gating_mechanism(self, context=None):

        # FIX 5/23/17: INTEGRATE THIS WITH ASSIGNMENT OF gating_signals
        # FIX:         (E.G., CHECK IF SPECIFIED GatingSignal ALREADY EXISTS)
        # Check the input_states and output_states of the System's Mechanisms
        #    for any GatingProjections with deferred_init()
        for mech in self.system.mechanisms:
            for state in mech._input_states + mech._output_states:
                for projection in state.mod_afferents:
                    # If projection was deferred for init, initialize it now and instantiate for self
                    if (projection.initialization_status == ContextFlags.DEFERRED_INIT
                        and projection.init_args['sender'] is None):
                        # FIX 5/23/17: MODIFY THIS WHEN (param, GatingProjection) tuple
                        # FIX:         IS REPLACED WITH (param, GatingSignal) tuple
                        # Add projection itself to any params specified in the GatingProjection for the GatingSignal
                        #    (cached in the GatingProjection's gating_signal attrib)
                        gating_signal_specs = projection.gating_signal or {}
                        gating_signal_specs.update({GATING_SIGNAL_SPECS: [projection]})
                        self._instantiate_gating_signal(gating_signal_specs, context=context)

        self._activate_projections_for_compositions(self.system)

    # Overrided gating_signals
    @property
    def gating_signals(self):
        try:
            return ContentAddressableList(component_type=GatingSignal,
                                          list=[state for state in self.output_states
                                                if isinstance(state, GatingSignal)])
        except:
            return None

    @gating_signals.setter
    def gating_signals(self, value):
        self._modulatory_signals = value

    # Suppress control_signals
    @property
    def control_signals(self):
        from psyneulink.core.components.mechanisms.adaptive.control import ControlMechanism
        from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
        raise GatingMechanismError(f"'control_signals' attribute is not implemented on {self.name} (a "
                                   f"{self.__class__.__name__}); consider using a {ControlMechanism.__name__} "
                                   f"instead, or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s "
                                   f"and {GatingSignal.__name__}s are needed.")

    @control_signals.setter
    def control_signals(self, value):
        from psyneulink.core.components.mechanisms.adaptive.control import ControlMechanism
        from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
        raise GatingMechanismError(f"'control_signals' attribute is not implemented on {self.name} (a "
                                   f"{self.__class__.__name__}); consider using a {ControlMechanism.__name__} "
                                   f"instead, or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s "
                                   f"and {GatingSignal.__name__}s are needed.")



# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _add_gating_mechanism_to_system(owner:GatingMechanism):

    if owner.gating_signals:
        for gating_signal in owner.gating_signals:
            for mech in [proj.receiver.owner for proj in gating_signal.efferents]:
                for system in mech.systems:
                    if owner not in system.execution_list:
                        system.execution_list.append(owner)
                        system.execution_graph[owner] = set()
                        # FIX: NEED TO ALSO ADD SystemInputState (AND ??ProcessInputState) PROJECTIONS
                        # # Add self to system's list of OriginMechanisms if it doesn't have any afferents
                        # if not any(state.path_afferents for state in owner.input_states):
                        #     system.origin_mechanisms.mechs.append(owner)
