# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  AGTControlMechanism ************************************************

"""

Contents
--------

  * `AGTControlMechanism_Overview`
  * `AGTControlMechanism_Creation`
  * `AGTControlMechanism_Structure`
      - `AGTControlMechanism_Input`
      - `AGTControlMechanism_Function`
      - `AGTControlMechanism_Output`
  * `AGTControlMechanism_Execution`
  * `AGTControlMechanism_Class_Reference`


.. _AGTControlMechanism_Overview:

Overview
--------

An AGTControlMechanism is a `ControlMechanism <ControlMechanism>` that uses an ObjectiveMechanism with a `DualAdaptiveIntegrator`
Function to regulate its `control_allocation <ControlMechanism.control_allocation>`.  When used with an `LCControlMechanism`
to regulate the `mode <FitzHughNagumoIntegrator.mode>` parameter of its `FitzHughNagumoIntegrator` Function, it implements a form of the
`Adaptive Gain Theory <http://www.annualreviews.org/doi/abs/10.1146/annurev.neuro.28.061604.135709>`_ of the locus
coeruleus-norepinephrine (LC-NE) system.

.. _AGTControlMechanism_Creation:

Creating an AGTControlMechanism
-------------------------------

An AGTControlMechanism can be created in any of the ways used to `create a ControlMechanism <ControlMechanism_Creation>`.

Like all ControlMechanisms, an AGTControlMechanism it receives its `input <AGTControlMechanism_Input>` from an `ObjectiveMechanism`.
However, unlike standard ControlMechanism, an AGTControlMechanism does not have an **objective_mechanism** argument in its
constructor.  When an AGTControlMechanism is created, it automatically creates an ObjectiveMechanism and assigns a
`DualAdaptiveIntegrator` Function as its `function <ObjectiveMechanism.function>`.

The OutputPorts to be monitored by the AGTControlMechanism's `objective_mechanism <AGTControlMechanism.objective_mechanism>` are
specified using the **monitored_output_ports** argument of the AGTControlMechanism's constructor, using any of the ways to
`specify the OutputPorts monitored by ObjectiveMechanism <ObjectiveMechanism_Monitor>`.  The
monitored OutputPorts are listed in the LCControlMechanism's `monitored_output_ports <AGTControlMechanism.monitored_output_ports>`
attribute,  as well as that of its `objective_mechanism <AGTControlMechanism.objective_mechanism>`.

The parameter(s) controlled by an AGTControlMechanism are specified in the **control_signals** argument of its constructor,
in the `standard way for a ControlMechanism <ControlMechanism_ControlSignals>`.

.. _AGTControlMechanism_Structure:

Structure
---------

.. _AGTControlMechanism_Input:

*Input: ObjectiveMechanism and Monitored OutputPorts*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An AGTControlMechanism has a single (primary) `InputPort <InputPort_Primary>` that receives its input via a
`MappingProjection` from the *OUTCOME* `OutputPort <ObjectiveMechanism_Output>` of an `ObjectiveMechanism`. The
ObjectiveMechanism is created automatically when the AGTControlMechanism is created, using a `DualAdaptiveIntegrator`
as its `function <ObjectiveMechanism.function>`, and is listed in the AGTControlMechanism's `objective_mechanism
<AGTControlMechanism.objective_mechanism>` attribute.  The ObjectiveMechanism aggregates the `value
<OutputPort.value>`\\s of the OutputPorts that it monitors, integrates their aggregated value at two different
rates, and combines those to generate the its output, which is used by the AGTControlMechanism as its input. The
OutputPorts monitored by the ObjectiveMechanism, listed in its `monitored_output_ports
<ObjectiveMechanism.monitored_output_ports>` attribute, are also listed in the AGTControlMechanism's
`monitored_output_ports <AGTControlMechanism_Base.monitored_output_ports>` attribute.  They can be displayed using
the AGTControlMechanism's `show <AGTControlMechanism.show>` method.

.. _AGTControlMechanism_Function:

*Function*
~~~~~~~~~~

An AGTControlMechanism uses the default function for a `ControlMechanism` (a default `Linear` Function), that simply passes
its input to its output.  Thus, it is the output of the AGTControlMechanism's `objective_mechanism
<AGTControlMechanism.objective_mechanism>` that determines its `control_allocation <ControlMechanism.control_allocation>`
and the `allocation <ControlSignal.allocation>` of its `ControlSignal(s) <ControlSignal>`.

.. _AGTControlMechanism_Output:

*Output*
~~~~~~~~

An AGTControlMechanism has a `ControlSignal` for each parameter specified in its `control_signals
<ControlMechanism.control_signals>` attribute, that sends a `ControlProjection` to the `ParameterPort` for the
corresponding parameter. ControlSignals are a type of `OutputPort`, and so they are also listed in the
AGTControlMechanism's `output_ports <AGTControlMechanism_Base.output_ports>` attribute. The parameters modulated by an
AGTControlMechanism's ControlSignals can be displayed using its `show <AGTControlMechanism_Base.show>` method. By default,
all of its ControlSignals are assigned the result of the AGTControlMechanism's `function <AGTControlMechanism.function>`, which is
the `input <AGTControlMechanism_Input>` it receives from its `objective_mechanism <AGTControlMechanism.objective_mechanism>`.
above).  The `allocation <ControlSignal.allocation>` is used by the ControlSignal(s) to determine
their `intensity <ControlSignal.intensity>`, which is then assigned as the `value <ControlProjection.value>` of the
ControlSignal's `ControlProjection`.   The `value <ControlProjection.value>` of the ControlProjection is used by the
`ParameterPort` to which it projects to modify the value of the parameter it controls (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter).

COMMENT:
FROM LCControlMechanism
If the **mode** argument of the LCControlMechanism's constructor is specified, the following Components are also
automatically created and assigned to the LCControlMechanism when it is created:

    * an `LCController` -- takes the output of the AGTUtilityIntegratorMechanism (see below) and uses this to
      control the value of the LCControlMechanism's `mode <FitzHughNagumoIntegrator.mode>` attribute.  It is assigned a single
      `ControlSignal` that projects to the `ParameterPort` for the LCControlMechanism's `mode <FitzHughNagumoIntegrator.mode>`
      attribute.
    ..
    * a `AGTUtilityIntegratorMechanism` -- monitors the `value <OutputPort.value>` of any `OutputPorts <OutputPort>`
      specified in the **mode** argument of the LCControlMechanism's constructor;  these are listed in the
      LCControlMechanism's `monitored_output_ports <LCControlMechanism.monitored_output_ports>` attribute,
      as well as that attribute of the AGTUtilityIntegratorMechanism and LCController.  They are evaluated by the
      AGTUtilityIntegratorMechanism's `DualAdaptiveIntegrator` Function, the result of whch is used by the LCControl to
      control the value of the LCControlMechanism's `mode <FitzHughNagumoIntegrator.mode>` attribute.
    ..
    * `MappingProjections <MappingProjection>` from Mechanisms or OutputPorts specified in **monitor_for_control** to
      the AGTUtilityIntegratorMechanism's `primary InputPort <InputPort_Primary>`.
    ..
    * a `MappingProjection` from the AGTUtilityIntegratorMechanism's *UTILITY_SIGNAL* `OutputPort
      <AGTUtilityIntegratorMechanism_Structure>` to the LCControlMechanism's *MODE* <InputPort_Primary>`.
    ..
    * a `ControlProjection` from the LCController's ControlSignal to the `ParameterPort` for the LCControlMechanism's
      `mode <FitzHughNagumoIntegrator.mode>` attribute.
COMMENT


.. _AGTControlMechanism_Execution:

Execution
---------

An AGTControlMechanism's `function <AGTControlMechanism_Base.function>` takes as its input the `value <InputPort.value>`
of its *OUTCOME* `input_port <Mechanism_Base.input_port>`, and uses that to determine its `control_allocation
<ITC.control_allocation>` which specifies the value assigned to the `allocation <ControlSignal.allocation>` of each of
its `ControlSignals <ControlSignal>`.  An AGTControlMechanism assigns the same value (the `input
<AGTControlMechanism_Input>` it receives from its `objective_mechanism <AGTControlMechanism.objective_mechanism>` to
all of its ControlSignals.  Each ControlSignal uses that value to calculate its `intensity <ControlSignal.intensity>`,
which is used by its `ControlProjection(s) <ControlProjection>` to modulate the value of the ParameterPort(s) for the
parameter(s) it controls, which are then used in the subsequent `TRIAL <TimeScale.TRIAL>` of execution.

.. note::
   A `ParameterPort` that receives a `ControlProjection` does not update its value until its owner Mechanism executes
   (see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).  This means that even if a
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the Mechanism to
   which it belongs has executed.


.. _AGTControlMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc

from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import DualAdaptiveIntegrator
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import MONITORED_OUTPUT_PORTS, ObjectiveMechanism
from psyneulink.core.components.shellclasses import Mechanism, System_Base
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.keywords import \
    CONTROL, CONTROL_PROJECTIONS, CONTROL_SIGNALS, INIT_EXECUTE_METHOD_ONLY, \
    MECHANISM, MULTIPLICATIVE, OBJECTIVE_MECHANISM
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'AGTControlMechanism', 'AGTControlMechanismError', 'ControlMechanismRegistry', 'MONITORED_OUTPUT_PORT_NAME_SUFFIX'
]

MONITORED_OUTPUT_PORT_NAME_SUFFIX = '_Monitor'

ControlMechanismRegistry = {}

class AGTControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class AGTControlMechanism(ControlMechanism):
    """
    AGTControlMechanism(                \
        monitored_output_ports=None,    \
        control_signals=None)

    Subclass of `ControlMechanism <ModulatoryMechanism>` that modulates the `multiplicative_param
    <Function_Modulatory_Params>` of the `function <Mechanism_Base.function>` of one or more `Mechanisms <Mechanism>`.
    See `ControlMechanism <ControlMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    monitored_output_ports : List[`OutputPort`, `Mechanism`, str, value, dict, `MonitoredOutputPortsOption`] or Dict
        specifies the OutputPorts to be monitored by the `objective_mechanism <AGTControlMechanism.objective_mechanism>`
        (see `monitored_output_ports <ObjectiveMechanism.monitored_output_ports>` for details of specification).

    control_signals : List[ParameterPort, tuple[str, Mechanism] or dict]
        specifies the parameters to be controlled by the AGTControlMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).


    Attributes
    ----------

    objective_mechanism : ObjectiveMechanism
        `ObjectiveMechanism` that monitors and evaluates the values specified in the ControlMechanism's
        **objective_mechanism** argument, the output of which is used as `input <AGTControlMechanism_Input>` to the
        AGTControlMechanism. It is created automatically when AGTControlMechanism is created, and uses as a
        `DualAdaptiveIntegrator` as is `function <ObjectiveMechanism.function>`.

    monitored_output_ports : List[OutputPort]
        each item is an `OutputPort` monitored by the `objective_mechanism <AGTControlMechanism.objective_mechanism>`; it is
        the same as the ObjectiveMechanism's `monitored_output_ports <ObjectiveMechanism.monitored_output_ports>`
        attribute. The `value <OutputPort.value>` of the OutputPorts listed are used by the ObjectiveMechanism to
        generate the AGTControlMechanism's `input <AGTControlMechanism_Input>`.

    monitored_output_ports_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding item of
        `monitored_output_ports <AGTControlMechanism.monitored_output_ports>`;  these are the same as those in
        the `monitored_output_ports_weights_and_exponents
        <ObjectiveMechanism.monitored_output_ports_weights_and_exponents>` attribute of the `objective_mechanism
        <AGTControlMechanism.objective_mechanism>`, and are used by the ObjectiveMechanism's `function
        <ObjectiveMechanism.function>` to parametrize the contribution made to its output by each of the values that
        it monitors (see `ObjectiveMechanism Function <ObjectiveMechanism_Function>`).

   """

    componentName = "AGTControlMechanism"

    initMethod = INIT_EXECUTE_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ControlMechanismClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    @tc.typecheck
    def __init__(self,
                 monitored_output_ports=None,
                 function=None,
                 # control_signals:tc.optional(tc.optional(list)) = None,
                 control_signals= None,
                 modulation:tc.optional(str)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        super().__init__(
            objective_mechanism=ObjectiveMechanism(
                monitored_output_ports=monitored_output_ports,
                function=DualAdaptiveIntegrator
            ),
            control_signals=control_signals,
            modulation=modulation,
            params=params,
            name=name,
            prefs=prefs,
        )

        self.objective_mechanism.name = self.name + '_ObjectiveMechanism'

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, MONITOR_FOR_CONTROL and CONTROL_SIGNALS

        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputPorts for Mechanisms in self.system
        Check that every item in `modulated_mechanisms <AGTControlMechanism.modulated_mechanisms>` is a Mechanism
            and that its function has a multiplicative_param
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if MONITORED_OUTPUT_PORTS in target_set and target_set[MONITORED_OUTPUT_PORTS] is not None:
            # It is a MonitoredOutputPortsOption specification
            if isinstance(target_set[MONITORED_OUTPUT_PORTS], MonitoredOutputPortsOption):
                # Put in a list (standard format for processing by _parse_monitored_output_ports_list)
                target_set[MONITORED_OUTPUT_PORTS] = [target_set[MONITORED_OUTPUT_PORTS]]
            # It is NOT a MonitoredOutputPortsOption specification, so assume it is a list of Mechanisms or Ports
            else:
                # Validate each item of MONITORED_OUTPUT_PORTS
                for item in target_set[MONITORED_OUTPUT_PORTS]:
                    if isinstance(item, MonitoredOutputPortsOption):
                        continue
                    if isinstance(item, tuple):
                        item = item[0]
                    if isinstance(item, dict):
                        item = item[MECHANISM]
                    if isinstance(item, (OutputPort, Mechanism)):
                        item = item.name
                    if not isinstance(item, str):
                        raise AGTControlMechanismError("Specification of {} arg for {} appears to be a list of "
                                                    "Mechanisms and/or OutputPorts to be monitored, but one"
                                                    "of the items ({}) is invalid".
                                                    format(OBJECTIVE_MECHANISM, self.name, item))
                    _parse_monitored_output_ports(source=self, output_port_list=item, context=context)

    @property
    def initial_short_term_utility(self):
        return self.objective_mechanism.function._initial_short_term_avg

    @initial_short_term_utility.setter
    def initial_short_term_utility(self, value):
        self.objective_mechanism.function.initial_short_term_avg = value

    @property
    def initial_long_term_utility(self):
        return self.objective_mechanism.function._initial_long_term_avg

    @initial_long_term_utility.setter
    def initial_long_term_utility(self, value):
        self.objective_mechanism.function.initial_long_term_avg = value

    @property
    def short_term_gain(self):
        return self.objective_mechanism.function._short_term_gain

    @short_term_gain.setter
    def short_term_gain(self, value):
        self.objective_mechanism.function.short_term_gain = value

    @property
    def long_term_gain(self):
        return self.objective_mechanism.function._long_term_gain

    @long_term_gain.setter
    def long_term_gain(self, value):
        self.objective_mechanism.function.long_term_gain = value

    @property
    def short_term_bias(self):
        return self.objective_mechanism.function._short_term_bias

    @short_term_bias.setter
    def short_term_bias(self, value):
        self.objective_mechanism.function.short_term_bias = value

    @property
    def long_term_bias(self):
        return self.objective_mechanism.function._long_term_bias

    @long_term_bias.setter
    def long_term_bias(self, value):
        self.objective_mechanism.function.long_term_bias = value

    @property
    def short_term_rate(self):
        return self.objective_mechanism.function._short_term_rate

    @short_term_rate.setter
    def short_term_rate(self, value):
        self.objective_mechanism.function.short_term_rate = value

    @property
    def long_term_rate(self):
        return self.objective_mechanism.function._long_term_rate

    @long_term_rate.setter
    def long_term_rate(self, value):
        self.objective_mechanism.function.long_term_rate = value

    @property
    def operation(self):
        return self.objective_mechanism.function._operation

    @operation.setter
    def operation(self, value):
        self.objective_mechanism.function.operation = value

    @property
    def agt_function_parameters(self):
        return self.objective_mechanism.function.parameters

    def show(self):
        """Display the `OutputPorts <OutputPort>` monitored by the AGTControlMechanism's `objective_mechanism`
        and the `multiplicative_params <Function_Modulatory_Params>` modulated by the AGTControlMechanism.
        """

        print("\n---------------------------------------------------------")

        print("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputPorts:")
        if self.objective_mechanism is None:
            print("\t\tNone")
        else:
            for port in self.objective_mechanism.input_ports:
                for projection in port.path_afferents:
                    monitored_port = projection.sender
                    monitored_port_Mech = projection.sender.owner
                    monitored_port_index = self.monitored_output_ports.index(monitored_port)

                    weight = self.monitored_output_ports_weights_and_exponents[monitored_port_index][0]
                    exponent = self.monitored_output_ports_weights_and_exponents[monitored_port_index][1]

                    print("\t\t{0}: {1} (exp: {2}; wt: {3})".
                          format(monitored_port_Mech.name, monitored_port.name, weight, exponent))

        print("\n\tModulating the following parameters:".format(self.name))
        # Sort for consistency of output:
        port_Names_sorted = sorted(self.output_ports.names)
        for port_Name in port_Names_sorted:
            for projection in self.output_ports[port_Name].efferents:
                print("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print("\n---------------------------------------------------------")
