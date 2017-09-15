# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  LCMechanism ************************************************

"""

.. note::
   **THIS MECHANISM IS ONLY PARTIALLY IMPLEMENTED.**

   IT CAN MODULATE MECHANISMS, BUT:

   - IT DOES NOT YET AUTOMATICALLY GENERATE A `LCController` OR `UtilityIntegrator` AS ITS OBJECTIVE MECHANISM
   ..
   - IT DOES NOT YET IMPLEMENT THe `FHNIntegrator` (FitzHugh-Nagumo) FUNCTION AND ASSOCIATED `mode` PARAMETER


Overview
--------

An LCMechanism is a `ControlMechanism <ControlMechanism>` that multiplicatively modulates the `function
<Mechanism_Base.function>` of one or more `Mechanisms <Mechanism>` (usually `TransferMechanisms <TransferMechanism>`).
It implements an abstract model of the `locus coeruleus (LC)  <https://www.ncbi.nlm.nih.gov/pubmed/12371518>`_ that,
together with a `UtilityIntegrator` Mechanism, implement a form of the `Adaptive Gain Theory
<http://www.annualreviews.org/doi/abs/10.1146/annurev.neuro.28.061604.135709>`_ of the locus coeruleus-norepinephrine
(LC-NE) system.  The LCMechanism uses an `FHNIntegrator` Function to generate its output, under the
influence of a `mode <LCMechanisms.mode>` parameter that regulates its functioning between its `"tonic" and "phasic"
modes of operation <LCMechanism_Modes_Of_Operation>`.  The Mechanisms modulated by an LCMechanism can be listed using
its `show <LCMechanism.show>` method.

.. _LCMechanism_Creation:

Creating an LCMechanism
-----------------------

An LCMechanism is generally created using its constructor.  The `inputs <LCMechanism_Input>` that drive its response
can be specified in its **input_states** argument, as a list of `Mechanisms <Mechanism>` and/or `OutputStates
<OutputState>` that should project to the LCMechanism.  The `Mechanisms <Mechanism>` it controls are specified in the
**modulated_mechanisms** argument of its constructor (see `LCMechanism_Modulate`).  It's **mode** argument is used to
specify either a default value for its `mode <LCMechanism.mode>` attribute, or another `ControlMechanism` used to
control it.

.. note::
   Unlike a standard `ControlMechanism`, an LCMechanism does not have an **objective_mechanism** argument in its
   constructor;  as noted above, its inputs are specified in its **input_states** argument.  However, if a list of
   OutputStates to monitor are specified in its **mode** argument, then a `ControlMechanism` and associated
   `ObjectiveMechanism` are created to monitor the `value <OutputState.value>` of the specified OutputStates and
   regulate the LCMechanism's `mode <LCMechanism.mode>` attribute.  In that case, the ObjectiveMechanism` is assigned
   as the LCMechanism's `objective_mechanism <LCMechanism.objective_mechanism>` attribute (see
   `LCMechanism_Monitored_OutputStates` and `LCMechanism_Modes_Of_Operation`), and the ControlMechanism as its
   `controller <LCMechanism.controller>` attribute.

.. _LCMechanism_Modulate:

Specifying Mechanisms to Modulate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Mechanisms to be modulated by an LCMechanism are specified in its **modulated_mechanisms** argument. An LCMechanism
controls a `Mechanism <Mechanism>` by modifying the `multiplicative_param <Function_Modulatory_Params>` of the
Mechanism's `function <TransferMechanism.function>`.  Therefore, any Mechanism specified for control by an LCMechanism
must be either a `TransferMechanism`, or a Mechanism that uses a `TransferFunction` or a class of `Function <Function>`
that implements a `multiplicative_param <Function_Modulatory_Params>`.  The **modulate_mechanisms** argument must be a
list of such Mechanisms.  The keyword *ALL* can also be used to specify all of the eligible `ProcessMechanisms
<ProcessingMechanism>` in all of the `Compositions <Composition>` to which the LCMechanism belongs.  If a Mechanism
specified in the **modulated_mechanisms** argument does not implement a multiplicative_param, it is ignored. A
`ControlProjection` is automatically created that projects from the LCMechanism to the `ParameterState` for the
`multiplicative_param <Function_Modulatory_Params>` of every Mechanism specified in the **modulated_mechanisms**
argument (and listed in its `modulated_mechanisms <LCMechanism.modulated_mechanisms>` attribute).


.. _LCMechanism_Monitored_OutputStates:

Specifying OutputStates to Monitor for Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the **mode** argument is specified in the LCMechanism's constructor, it automatically creates an `LCController` (as
its `ControlMechanism`) and associated `UtilityIntegratorMechanism` (as its `ObjectiveMechanism`) that are used to
monitor and evaluate the `value <OutputState.value>` of the `OutputStates <OutputState>` specified, and regulate the
value of the LCMechanism's `mode <LCMechanism.mode>` attribute (see `LCMechanism_Modes_Of_Operation`). The **mode**
argument must be a list, each item of which must refer to a `Mechanism <Mechanism>` or the `OutputState` of one.
These are assigned to the `monitored_output_states <UtilityIntegratorMechanism>` attribute of the
UtilityIntegratorMechanism, and to the same attribute of the LCController and the LCMechanism itself. The
UtilityIntegratorMechanism is assigned to the `objective_mechanism <LCController.objective_mechanism>` attribute,
as well as that of the LCMechanism.  Finally, the LCController is assigned as the `controller
<LCMechanism.controller>` attribute of the LCMechanism.

.. _LCMechanism_Structure:

Structure
---------

.. _LCMechanism_Input:

Input
~~~~~

An LCMechanism has a single (primary) `InputState <InputState_Primary>` that receives projections from any Mechanisms
specified in the **input_states** argument of the LCMechanism's constructor;  its `value <InputState.value>` is used as
the input to the LCMechanism's `function <LCMechanism.function>`.

.. _LCMechanism_Function:

Function
~~~~~~~~

An LCMechanism uses the `FHNIntegrator` as its `LCMechanism.function`; this implements a `FitzHugh-Nagumo model
<https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model>`_ often used to describe the spiking of a neuron, but in this
case the population activity of the LC (see `Gilzenrat et al., <2002https://www.ncbi.nlm.nih.gov/pubmed/12371518>`_).
The `FNH` Function takes the `input <LCMechanism_Input>` to the LCMechanism as its `variable <FHN.variable>`,
and uses the LCMechanism's `mode <LCMechanism.mode>` attribute as its `mode <FHN.mode>` attribute.

.. _LCMechanism_Modes_Of_Operation:

LC Modes of Operation
^^^^^^^^^^^^^^^^^^^^^
The LCMechanism's `mode <LCMechanism.mode>` attribute regulates its function between `"tonic" and "phasic" modes
of operation <https://www.ncbi.nlm.nih.gov/pubmed/8027789>`_:

  * in the *tonic mode* (low value of `mode <LCMechanism.mode>`), the output of the LCMechanism is moderately low
    and constant; that is, it is relatively unaffected by its `input <LCMechanism_Input`.  This blunts the response
    of the Mechanisms that the LCMechanism controls to their inputs.

  * in the *phasic mode* (high value of `mode <LCMechanism.mode>`), when the `input to the LC <LC_Input>` is low,
    its `output <LC_Ouput>` is even lower than when it is in the tonic regime, and thus the response of the
    Mechanisms it controls to their outputs is even more blunted.  However, when the LCMechanism's input rises above
    a certain value (determined by the `threshold <LCMechanism.threshold>` parameter), its output rises sharply,
    producing a much sharper response of the Mechanisms it controls to their inputs.

If the **mode** argument of the LCMechanism's constructor is specified, the following Components are also
automatically created and assigned to the LCMechanism when it is created:

    * an `LCController` -- takes the output of the UtilityIntegratorMechanism (see below) and uses this to
      control the value of the LCMechanism's `mode <LCMechanism.mode>` attribute.  It is assigned a single
      `ControlSignal` that projects to the `ParameterState` for the LCMechanism's `mode <LCMechanism.mode>` attribute.
    ..
    * a `UtilityIntegratorMechanism` -- monitors the `value <OutputState.value>` of any `OutputStates <OutputState>`
      specified in the **mode** argument of the LCMechanism's constructor;  these are listed in the LCMechanism's
      `monitored_output_states <LCMechanism.monitored_output_states>` attribute, as well as that attribute of the
      UtilityIntegratorMechanism and LCController.  They are evaluated by the UtilityIntegratorMechanism's
      `UtilityIntegrator` Function, the result of whch is used by the LCControl to control the value of the
      LCMechanism's `mode <LCMechanism.mode>` attribute.
    ..
    * `MappingProjections <MappingProjection>` from Mechanisms or OutputStates specified in **monitor_for_control** to
      the UtilityIntegratorMechanism's `primary InputState <InputState_Primary>`.
    ..
    * a `MappingProjection` from the UtilityIntegratorMechanism's *UTILITY_SIGNAL* `OutputState
      <UtilityIntegratorMechanism_Structure>` to the LCMechanism's *MODE* <InputState_Primary>`.
    ..
    * a `ControlProjection` from the LCController's ControlSignal to the `ParameterState` for the LCMechanism's
      `mode <LCMechanism.mode>` attribute.

.. _LCMechanism_Output:

Output
~~~~~~

COMMENT:
VERSION FOR SINGLE ControlSignal
An LCMechanism has a single `ControlSignal` used to modulate the function of the Mechanism(s) listed in its
`modulated_mechanisms <LCMechanism.modulated_mechanisms>` attribute.  The ControlSignal is assigned a
`ControlProjection` to the `ParameterState` for the `multiplicative_param <Function_Modulatory_Params>` of the
`function <Mechanism_Base.function>` for each of those Mechanisms.
COMMENT

An LCMechanism has a `ControlSignal` for each Mechanism listed in its `modulated_mechanisms
<LCMechanism.modulated_mechanisms>` attribute.  All of its ControlSignals are assigned the same value:  the result of
the LCMechanism's `function <LCMechanism.function>`.  Each ControlSignal is assigned a `ControlProjection` to the
`ParameterState` for the  `multiplicative_param <Function_Modulatory_Params>` of `function
<Mechanism_Base.function>` for the Mechanism in `modulated_mechanisms <LCMechanism.modulate_mechanisms>` to which it
corresponds. The Mechanisms modulated by an LCMechanism can be displayed using its :func:`show <LCMechanism.show>`
method.

.. _LCMechanism_Examples:

Examples
~~~~~~~~

The following example generates an LCMechanism that modulates the function of two TransferMechanisms, one that uses
a `Linear` function and the other a `Logistic` function::

    my_mech_1 = TransferMechanism(function=Linear,
                                  name='my_linear_mechanism')
    my_mech_2 = TransferMechanism(function=Logistic,
                                  name='my_logistic_mechanism')

    LC = LCMechanism(modulated_mechanisms=[my_mech_1, my_mech_2],
                     name='my_LC')

Calling `my_LC.show()` generates the following report::

    my_LC
COMMENT:
        Monitoring the following Mechanism OutputStates:
            None
COMMENT

        Modulating the following Mechanism parameters:
            my_logistic_mechanism: gain
            my_linear_mechanism: slope

Note that the LCMechanism controls the `multiplicative_param <Function_Modulatory_Params>` of the `function
<Mechanism_Base.function>` of each Mechanism:  the `gain <Logistic.gain>` parameter for ``my_mech_1``, since it uses
a `Logistic` Function; and the `slope <Linear.slope>` parameter for ``my_mech_2``, since it uses a `Linear` Function.

COMMENT:

ADDITIONAL EXAMPLES HERE OF THE DIFFERENT FORMS OF SPECIFICATION FOR
**monitor_for_control** and **modulated_mechanisms**

STRUCTURE:
MODE INPUT_STATE <- NAMED ONE, LAST?
SIGNAL INPUT_STATE(S) <- PRIMARY;  MUST BE FROM PROCESSING MECHANISMS
CONTROL SIGNALS

COMMENT

.. _LCMechanism_Execution:

Execution
---------

An LCMechanism executes within a `Composition` at a point specified in the Composition's `Scheduler` or, if it is
the `controller <System_Base>` for a `Composition`, after all of the other Mechanisms in the Composition have `executed
<Composition_Execution>` in a `TRIAL`. It's `function <LCMechanism.function>` takes the `value <InputState.value>` of
the LCMechanism's `primary InputState <InputState_Primary>` as its input, and generates a response -- under the
influence of its `mode <LCMechanism.mode>` parameter -- that is assigned as the `value <ControlSignal.value>` of its
`ControlSignals <ControlSignal>`.  The latter are used by its `ControlProjections <ControlProjection>` to modulate the
response -- in the next `TRIAL` of execution --  of the Mechanisms the LCMechanism controls.

.. note::
   A `ParameterState` that receives a `ControlProjection` does not update its value until its owner Mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   LCMechanism has executed, the `multiplicative_param <Function_Modulatory_Params>` parameter of the `function
   <Mechanism_Base.function>` of a Mechanism that it controls will not assume its new value until that Mechanism has
   executed.

.. _LCMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc

from PsyNeuLink.Components.Functions.Function import ModulationParam, _is_modulation_param, MULTIPLICATIVE_PARAM
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.ControlMechanism \
    import ControlMechanism_Base, ALLOCATION_POLICY
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.Functions.Function import Integrator
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.ShellClasses import Mechanism
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Keywords import FUNCTION, ALL, INIT__EXECUTE__METHOD_ONLY, INPUT_STATES, \
                                        CONTROL_PROJECTIONS, CONTROL_SIGNALS

from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

MODULATED_MECHANISMS = 'modulated_mechanisms'
CONTROL_SIGNAL_NAME = 'LCMechanism_ControlSignal'

ControlMechanismRegistry = {}

class LCMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class LCMechanism(ControlMechanism_Base):
    """
    LCMechanism(                               \
    input_states=None,                         \
    mode=0.0,                                  \
    modulated_mechanisms=None,                 \
    params=None,                               \
    name=None,                                 \
    prefs=None)

    Subclass of `ControlMechanism <AdaptiveMechanism>` that modulates the `multiplicative_param
    <Function_Modulatory_Params>` of the `function <Mechanism_Base.function>` of one or more `Mechanisms <Mechanism>`.

    Arguments
    ---------

    input_states :  List[`OutputState`, `Mechanism`] : default None
        specifies the OutputStates from which the LCMechanism should receive its `input <LCMechanism_Input>`;  if a
        Mechanism is specified, its `primary OutputState <OutputState_Primary>` is used. A `MappingProjection` is
        generated from each OutputState specified to a corresponding InputState created for the LCMechanism.

    mode : float or List[List[monitored_output_states specification]: default 0.0
        specifies the default value for the `mode <FHNIntegrator.mode>` parameter of the LCMechanism's `function
        <LCMechanism.function>`, or a list of OutputStates to be monitored, the values of which are used to regulate
        the value of the `mode <FHNIntegrator.mode>` parameter.  It can be specified using any of the forms used to
        `specify the <ObjectiveMechanism_Monitored_output_states>` **monitored_output_states** `argument
        <ObjectiveMechanism_Monitored_output_states>` of an ObjectiveMechanism.

    modulated_mechanisms : List[`Mechanism`] or *ALL*
        specifies the Mechanisms to be modulated by the LCMechanism. If it is a list, every item must be a Mechanism
        with a `function <Mechanism_Base.function>` that implements a `multiplicative_param
        <Function_Modulatory_Params>`;  alternatively the keyword *ALL* can be used to specify all of the
        `ProcessingMechanisms <ProcessingMechanism>` in the Composition(s) to which the LCMechanism
        belongs.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default LCMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for the Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    input_states : ContentAddressableList[`InputState`]
        a list of the LCMechanism's `InputStates <Mechanism_InputStates>` that determine its `input
        <LCMechanism_Input>`;  each receives a `MappingProjection` from an `OutputState` specified in the
        **input_states** argument of the LCMechanism's constructor. The sum of their `value <InputState.value>`\\s is
        used as the `variable <FHNIntegrator.variable>` for the LCMechanism's `function <LCMechanism.function>`.

    mode : float : default 0.0
        determines the value for the `mode <FHNIntegrator.mode>` parameter of the LCMechanism's
        `FHNIntegrator` function (see `LCMechanism_Modes_Of_Operation` for additional details).

    controller : None or LCController
        lists the ControlMechanism used to regulate the value of the LCMechanism's `mode <LCMechanism.mode>` attribute.
        It receives its input from the LCMechanism's `objective_mechanism`, and sends a `ControlProjection` to its
        `primary InputState <InputState_Primary>`.

    objective_mechanism : None or `ObjectiveMechanism`
        lists the Mechanism used to monitor and evaluate any OutputStates specified in the **mode** argument of the
        LCMechanism's constructor, the `value <OutputState.value>`\\s of which are used by its `controller
        <LCMechanism.controller>` to regulate the value of its `mode <LCMechanism.mode>` attribute.  It is the same
        as the `controller <LCMechanism.controller>`'s `objective_mechanism <LCController.objective_mechanism>`
        attribute.

    monitored_output_states : None or List[`OutputState`]
        lists the `OutputStates <OutputState>` specified in the **mode** argument of the LCMechanism's constructor,
        and monitored by its `objective_mechanism <LCMechanism.objective_mechanism>`; their `value
        <OutputState.value>`\\s are used by the LCMechanism's `controller <LCMechanism.controller>` to regulate
        the value of its `mode <LCMechanism.mode>` attribute.  It is the same as the :keyword:`monitored_output_states`
        attributes of the `controller <LCMechanism.controller>` and `objective_mechanism
        <LCMechanism.objective_mechanism>`.

    function : `FitzHughNagumoIntegrator`
        takes the LCMechanism's `input <LCMechanism_Input>` and generates its response <LCMechanism_Output>` under
        the influence of its `mode <LCMechanism.mode>` attribute (see `LCMechanism_Function` for additional details).

    COMMENT:
    VERSIONS FOR SINGLE ControlSignal
        control_signals : List[ControlSignal]
            contains the LCMechanism's single `ControlSignal`, which sends `ControlProjections` to the
            `multiplicative_param <Function_Modulatory_Params>` of each of the Mechanisms the LCMechanism
            controls (listed in its `modulated_mechanisms <LCMechanism.modulated_mechanisms>` attribute).

        control_projections : List[ControlProjection]
            list of `ControlProjections <ControlProjection>` sent by the LCMechanism's `ControlSignal`, each of which
            projects to the `ParameterState` for the `multiplicative_param <Function_Modulatory_Params>` of the
            `function <Mechanism_Base.function>` of one of the Mechanisms listed in `modulated_mechanisms
            <LCMechanism.modulated_mechanisms>` attribute.
    COMMENT

    control_signals : List[`ControlSignal`]
        contains a ControlSignal for each Mechanism listed in the LCMechanism's `modulated_mechanisms
        <LCMechanism.modulated_mechanisms>` attribute; each ControlSignal sends a `ControlProjections` to the
        `ParameterState` for the `multiplicative_param <Function_Modulatory_Params>` of the `function
        <Mechanism_Base.function>corresponding Mechanism.

    control_projections : List[`ControlProjection`]
        list of all of the `ControlProjections <ControlProjection>` sent by the `ControlSignals <ControlSignal>` listed
        in `control_signals <LC_Mechanism.control_signals>`.

    modulated_mechanisms : List[`Mechanism`]
        list of Mechanisms modulated by the LCMechanism.

    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE
        the default form of modulation used by the LCMechanism's `ControlProjections`,
        unless they are `individually specified <ControlSignal_Specification>`.

    """

    componentType = "LCMechanism"

    initMethod = INIT__EXECUTE__METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ControlMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    class ClassDefaults(AdaptiveMechanism_Base.ClassDefaults):
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = ControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({FUNCTION:Integrator,
                               CONTROL_SIGNALS: None,
                               CONTROL_PROJECTIONS: None,
                               })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 monitor_for_control:tc.optional(list)=None,
                 mode:tc.optional(float)=0.0,
                 modulated_mechanisms:tc.optional(tc.any(list,str)) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mode=mode,
                                                  modulated_mechanisms=modulated_mechanisms,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         size=size,
                         monitor_for_control=monitor_for_control,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, MONITOR_FOR_CONTROL and CONTROL_SIGNALS

        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputStates for Mechanisms in self.system
        Check that every item in `modulated_mechanisms <LCMechanism.modulated_mechanisms>` is a Mechanism
            and that its function has a multiplicative_param
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        # if MONITOR_FOR_CONTROL in target_set:
        #     for spec in target_set[MONITOR_FOR_CONTROL]:
        #         if isinstance(spec, MonitoredOutputStatesOption):
        #             continue
        #         if isinstance(spec, tuple):
        #             spec = spec[0]
        #         if isinstance(spec, (OutputState, Mechanism_Base)):
        #             spec = spec.name
        #         if not isinstance(spec, str):
        #             raise LCMechanismError("Invalid specification in {} arg for {} ({})".
        #                                         format(MONITOR_FOR_CONTROL, self.name, spec))
        #         # If controller has been assigned to a System,
        #         #    check that all the items in monitor_for_control are in the same System
        #         # IMPLEMENTATION NOTE:  If self.system is None, onus is on doing the validation
        #         #                       when the controller is assigned to a System [TBI]
        #         if self.system:
        #             if not any((spec is mech.name or spec in mech.output_states.names)
        #                        for mech in self.system.mechanisms):
        #                 raise LCMechanismError("Specification in {} arg for {} ({}) must be a "
        #                                             "Mechanism or an OutputState of one in {}".
        #                                             format(MONITOR_FOR_CONTROL, self.name, spec, self.system.name))

        if MODULATED_MECHANISMS in target_set and target_set[MODULATED_MECHANISMS]:

            from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal

            spec = target_set[MODULATED_MECHANISMS]

            if isinstance (spec, str):
                if not spec == ALL:
                    raise LCMechanismError("A string other than the keyword \'ALL\' was specified for the {} argument "
                                           "the constructor for {}".format(MODULATED_MECHANISMS, self.name))

            if not isinstance(spec, list):
                spec = [spec]

            for mech in spec:
                if not isinstance(mech, Mechanism):
                    raise LCMechanismError("The specification of the {} argument for {} contained an item ({})"
                                           "that is not a Mechanism.".format(MODULATED_MECHANISMS, self.name, mech))

                if not hasattr(mech.function_object, MULTIPLICATIVE_PARAM):
                    raise LCMechanismError("The specification of the {} argument for {} contained a Mechanism ({})"
                                           "that does not have a {}.".
                                           format(MODULATED_MECHANISMS, self.name, mech, MULTIPLICATIVE_PARAM))

    def _instantiate_input_states(self, context=None):
        """Instantiate input_value attribute

        Instantiate input_states and monitored_output_states attributes (in case they are referenced)
            and assign any OutputStates that project to the input_states to monitored_output_states

        IMPLEMENTATION NOTE:  At present, these are dummy assignments, simply to satisfy the requirements for
                              subclasses of ControlMechanism;  in the future, an _instantiate_objective_mechanism()
                              method should be implemented that also implements an _instantiate_monitored_output_states
                              method, and that can be used to add OutputStates/Mechanisms to be monitored.
        """

        self.monitored_output_states = []

        if not hasattr(self, INPUT_STATES):
            self._input_states = None
        elif self.input_states:
            for input_state in self.input_states:
                for projection in input_state.path_afferents:
                    self.monitored_output_states.append(projection.sender)



    def _instantiate_output_states(self, context=None):
        """Instantiate ControlSignal and assign ControlProjections to Mechanisms in self.modulated_mechanisms

        If **modulated_mechanisms** argument of constructor was specified as *ALL*,
            assign all ProcessingMechanisms in Compositions to which LCMechanism belongs to self.modulated_mechanisms
        Instantiate ControlSignal with Projections to the ParameterState for the multiplicative_param of every
           Mechanism listed in self.modulated_mechanisms

        Returns ControlSignal (OutputState)
        """
        from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
        from PsyNeuLink.Components.States.ParameterState import _get_parameter_state
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base

        # *ALL* is specified for modulated_mechanisms:
        #    assign all Processing Mechanisms in the LCMechanism's Composition(s) to its modulated_mechanisms attribute
        if isinstance(self.modulated_mechanisms, str) and self.modulated_mechanisms is ALL:
            self.modulated_mechanisms = []
            for system in self.systems:
                for mech in system.mechanisms:
                    if isinstance(mech, ProcessingMechanism_Base) and hasattr(mech.function, MULTIPLICATIVE_PARAM):
                            self.modulated_mechanisms.append(mech)
            for process in self.processes:
                for mech in process.mechanisms:
                    if isinstance(mech, ProcessingMechanism_Base) and hasattr(mech.function, MULTIPLICATIVE_PARAM):
                            self.modulated_mechanisms.append(mech)

        # # MODIFIED 9/3/17 OLD [ASSIGN ALL ControlProjections TO A SINGLE ControlSignal]
        # # Get the ParameterState for the multiplicative_param of each Mechanism in self.modulated_mechanisms
        # multiplicative_params = []
        # for mech in self.modulated_mechanisms:
        #     multiplicative_params.append(mech._parameter_states[mech.function_object.multiplicative_param])
        #
        # # Create specification for **control_signals** argument of ControlSignal constructor
        # self.control_signals = [{CONTROL_SIGNAL_NAME:multiplicative_params}]

        # MODIFIED 9/3/17 NEW [ASSIGN EACH ControlProjection TO A DIFFERENT ControlSignal]
        # Get the name of the multiplicative_param of each Mechanism in self.modulated_mechanisms
        multiplicative_param_names = []
        for mech in self.modulated_mechanisms:
            multiplicative_param_names.append(mech.function_object.multiplicative_param)

        # Create specification for **control_signals** argument of ControlSignal constructor
        self.control_signals = []
        for mech, mult_param_name in zip(self.modulated_mechanisms, multiplicative_param_names):
            self.control_signals.append((mult_param_name, mech))

        # MODIFIED 9/3/17 END

        super()._instantiate_output_states(context=context)

    # def _instantiate_attributes_after_function(self, context=None):
    #     """Implement ControlSignals specified in control_signals arg or "locally" in parameter specification(s)
    #
    #     Calls super's instantiate_attributes_after_function, which calls _instantiate_output_states;
    #         that insures that any ControlSignals specified in control_signals arg are instantiated first
    #     Then calls _assign_as_controller to instantiate any ControlProjections/ControlSignals specified
    #         along with parameter specification(s) (i.e., as part of a (<param value>, ControlProjection) tuple
    #     """
    #
    #     super()._instantiate_attributes_after_function(context=context)
    #
    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Updates LCMechanism's ControlSignal based on input and mode parameter value
        """
        return self.function()

    @tc.typecheck
    def add_modulated_mechanisms(self, mechanisms:list):
        """Add ControlProjections to the specified Mechanisms.
        """

        request_set = {MODULATED_MECHANISMS:mechanisms}
        target_set = {}
        self._validate_params(request_set=request_set, target_set=target_set)

        # Assign ControlProjection from the LCMechanism's ControlSignal
        #    to the ParameterState for the multiplicative_param of each Mechanism in mechanisms
        multiplicative_params = []
        for mech in mechanisms:
            self.modulated_mechanisms.append(mech)
            parameter_state = mech._parameter_states[mech.multiplicative_param]
            control_projection = ControlProjection(sender=self.control_signals[0],
                                                   receiver=parameter_state)
            self.control_projections.append(control_projection)

    @tc.typecheck
    def remove_modulated_mechanisms(self, mechanisms:list):
        """Remove the ControlProjections to the specified Mechanisms.
        """

        for mech in mechanisms:
            if not mech in self.modulated_mechanisms:
                continue

            parameter_state = mech._parameter_states[mech.multiplicative_param]

            # Get ControlProjection
            for projection in parameter_state.mod_afferents:
                if projection.sender.owner is self:
                    control_projection = projection
                    break

            # Delete ControlProjection ControlSignal's list of efferents
            index = self.control_signals[0].efferents[control_projection]
            del(self.control_signals[0].efferents[index])

            # Delete ControlProjection from recipient ParameterState
            index = parameter_state.mod_afferents[control_projection]
            del(parameter_state.mod_afferents[index])

            # Delete ControlProjection from self.control_projections
            index = self.control_projections[control_projection]
            del(self.control_projections[index])

            # Delete ControlProjection
            del(control_projection)

            # Delete Mechanism from self.modulated_mechanisms
            index = self.modulated_mechanisms.index(mech)
            del(self.modulated_mechanisms[index])

    def show(self):
        """Display the `OutputStates <OutputState>` monitored by the LCMechanism's `objective_mechanism`
        and the `multiplicative_params <Function_Modulatory_Params>` modulated by the LCMechanism.
        """

        print ("\n---------------------------------------------------------")

        print ("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputStates:")
        if self.objective_mechanism is None:
            print ("\t\tNone")
        else:
            for state in self.objective_mechanism.input_states:
                for projection in state.path_afferents:
                    monitored_state = projection.sender
                    monitored_state_mech = projection.sender.owner
                    monitored_state_index = self.monitored_output_states.index(monitored_state)

                    weight = self.monitored_output_states_weights_and_exponents[monitored_state_index][0]
                    exponent = self.monitored_output_states_weights_and_exponents[monitored_state_index][1]

                    print ("\t\t{0}: {1} (exp: {2}; wt: {3})".
                           format(monitored_state_mech.name, monitored_state.name, weight, exponent))

        print ("\n\tModulating the following parameters:".format(self.name))
        # Sort for consistency of output:
        state_names_sorted = sorted(self.output_states.names)
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].efferents:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")
