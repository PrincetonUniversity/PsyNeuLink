# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  GatingSignal *****************************************************

"""

Contents:
---------

  * `Gating_Signal_Overview`
  * `GatingSignal_Creation`
      - `GatingSignal_Specification`
  * `GatingSignal_Structure`
      - `GatingSignal_Projections`
      - `GatingSignal_Modulation`
  * `GatingSignal_Execution`
  * `GatingSignal_Examples`
  * `GatingSignal_Class_Reference`


.. _Gating_Signal_Overview:

Overview
--------

A GatingSignal is a type of `ModulatorySignal <ModulatorySignal>` that is specialized for use with a `GatingMechanism`
and one or more `GatingProjections <GatingProjection>`, to modify the `value <Port_Base.value>`\\(s) of the
`InputPort(s) <InputPort>` and/or `OutputPort(s) <OutputPort>` to which they project. A GatingSignal receives the
value from the `gating_allocation <GatingMechanism.gating_allocation>` of the GatingMechanism to which it belongs,
and assigns that as the value of its `gating_signal <GatingSignal.gating_signal>` to its `GatingProjection(s)
<GatingProjection>`, each of which projects to an InputPort or OutputPort and is used to modulate the `value
<Port_Base.value>` of that Port.


.. _GatingSignal_Creation:

Creating a GatingSignal
-----------------------

A GatingSignal is created automatically whenever an `InputPort` or `OutputPort` of a `Mechanism <Mechanism>` is
specified for gating.  This can be done either in the **gate** argument of the constructor for a `GatingMechanism
<GatingMechanism_GatingSignals>`, or in the `specification of projections <Port_Projections>` to the InputPort or
OutputPort.  Although a GatingSignal can be created directly using its constructor (or any of the other ways for
`creating an OutputPort <OutputPorts_Creation>`), this is usually not necessary nor is it advisable, as a GatingSignal
has dedicated components and requirements for configuration that must be met for it to function properly.

.. _GatingSignal_Specification:

*Specifying GatingSignals*
~~~~~~~~~~~~~~~~~~~~~~~~~~

When a GatingSignal is specified in the **gate** argument of the constructor for a `GatingMechanism`, the
InputPort(s) and/or OutputPort(s) it gates must be specified. This can take any of the following forms:

  * **InputPort** or **OutputPort** of a Mechanism;
  ..
  * **Mechanism** -- the `primary `InputPort <InputPort_Primary>` or `OutputPort <OutputPort_Primary>` is used;
  ..
  * **specification dictionary** -- can take either of the following two forms:

    * for gating a single port, the dictionary can have the following two entries:

        * *NAME*: str
            the string must be the name of the Port to be gated; the GatingSignal will named by appending
            "_GatingSignal" to the name of the Port.

        * *MECHANISM*: Mechanism
            the Mechanism must be the one to the which the Port to be gated belongs.

    * for gating multiple ports, the dictionary can have the following entry:

        * <str>:list
            the string used as the key specifies the name to be used for the GatingSignal,
            and each item of the list must be a `specification of a Port <State_Creation>` to be
            gated by the GatingSignal (and that will receive a `GatingProjection` from it).

    The dictionary can also contain entries for any other GatingSignal attributes to be specified
    (e.g., a *MODULATION* entry, the value of which determines how the GatingSignal modulates the
    `value <Port_Base.value>` of the Port(s) that it gates; or a *VARIABLE* entry specifying which item
    of the GatingMechanism's `gating_allocation <GatingMechanism.gating_allocation>` it should use as its `value
    <GatingSignal,value>`;  see `OutputPort_Customization`).
  ..
  * **2-item tuple:** *(<Port name or list of Port names>, <Mechanism>)* -- the 1st item must be the name of the
    Port (or list of Port names), and the 2nd item the Mechanism to which it (they) belong(s); this is a convenience
    format, which is simpler to use than a specification dictionary (see below), but precludes specification of
    `parameters <GatingSignal_Structure>` for the GatingSignal.

.. _GatingSignal_Structure:

Structure
---------

A GatingSignal is owned by a `GatingMechanism`, and associated with one or more `GatingProjections <GatingProjection>`,
each of which projects to the InputPort or OutputPort that it gates.

.. _GatingSignal_Projections:

*Projections*
~~~~~~~~~~~~~

When a GatingSignal is created, it can be assigned one or more `GatingProjections <GatingProjection>`, using either
the **projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument
with the key *PROJECTIONS*.  These will be assigned to its `efferents <GatingSignal.efferents>` attribute.  See
`Port Projections <Port_Projections>` for additional details concerning the specification of Projections when
creating a Port.

.. note::
   Although a GatingSignal can be assigned more than one `GatingProjection`, all of those Projections will convey
   the same `gating_signal <LearningMechanism>` (received from the GatingMechanism to which the GatingSignal
   belongs), and use the same form of `modulation <GatingSignal_Modulation>`.  This can be useful for implementing
   broadly projecting modulatory effects.

.. _GatingSignal_Modulation:

*Modulation*
~~~~~~~~~~~~

Each GatingSignal has a `modulation <GatingSignal.modulation>` attribute that determines how the GatingSignal's
`value <GatingSignal.value>` (i.e., its `gating_signal <GatingSignal.gating_signal>`) is used by the Ports to which it
projects to modify their `value <Port_Base.value>` \\s (see `ModulatorySignal_Modulation` for an explanation of how the
`modulation <GatingSignal.modulation>` attribute is specified and used to modulate the `value <Port_Base.value>` of a
Port). The `modulation <GatingSignal.modulation>` attribute can be specified in the **modulation** argument of the
constructor for a GatingSignal, or in a specification dictionary as described `above <GatingSignal_Specification>`.
If it is not specified, its default is the value of the `modulation <GatingMechanism.modulation>` attribute of the
GatingMechanism to which the GatingSignal belongs (which is the same for all of the GatingSignals belonging to that
GatingMechanism).  The value of the `modulation <GatingSignal.modulation>` attribute of a GatingSignal is used by all
of the `GatingProjections <GatingProjection>` that project from that GatingSignal.


.. _GatingSignal_Execution:

Execution
---------

A GatingSignal cannot be executed directly.  It is executed whenever the `GatingMechanism` to which it belongs is
executed.  When this occurs, the GatingMechanism provides the GatingSignal with one of the values from its
`gating_allocation <GatingMechanism.gating_signal>`, that is used by its `function <GatingSignal.function>` to generate its
the value of its `gating_signal <GatingSignal.gating_signal>`.  That, in turn, is used by its `GatingProjection(s)
<GatingProjection>` to modulate the `value <Port_Base.value>` of the Ports to which they project. How the modulation
is executed is determined by the GatingSignal's `modulation <GatingSignal.modulation>` attribute
(see `above <GatingSigna_Modulation>`, and `ModulatorySignal_Modulation` for a more detailed explanation of how
modulation operates).

.. note::
   The change in the `value <Port_Base.value>` of InputPorts and OutputPorts in response to the execution of a
   GatingSignal are not applied until the Mechanism(s) to which those ports belong are next executed;
   see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).

.. _GatingSignal_Examples:

Examples
--------

**Gate an InputPort and OutputPort**.  In the following example, `GatingMechanism` is configured to gate the
`primary InputPort <InputPort_Primary>` of one Mechanism, and the `primary OutputPort <OutputPort_Primary>`
of another::

    >>> import psyneulink as pnl
    >>> my_mechanism_A = pnl.TransferMechanism(name="Mechanism A")
    >>> my_mechanism_B = pnl.TransferMechanism(name="Mechanism B")
    >>> my_gating_mechanism = pnl.GatingMechanism(gate=[my_mechanism_A.input_port,
    ...                                                 my_mechanism_B.output_port])

Note that, in the **gate** argument, the first item references a Mechanism (``my_mechanism_A``) rather than
one of its ports -- this is all that is necessary, since the default for a `GatingSignal` is to modulate the
`primary InputPort <InputPort_Primary>` of a Mechanism.  The second item explicitly specifies the Port to be gated,
since it is an OutputPort.  This will generate two GatingSignals, each of which will multiplicatively modulate the
value of the Port to which it projects.  This is because, by default, the `modulation <GatingSignal.modulation>`
attribute of a GatingSignal is the *MULTIPLICATIVE_PARAM* for the `function <Port_Base.function>` of the Port to which
it projects.  For an InputPort, the default `function <InputPort.function>` is `Linear` and its *MULTIPLICATIVE_PARAM*
is its `slope <Linear.slope>` parameter.  Thus, the value of the GatingSignal is assigned to the slope, which multiplies
the Port`s `variable <Port_Base.variable>` (i.e., its input(s)) to determine its `value <Port_Base.value>`.

**Modulate the InputPorts of several Mechanisms**.  In next example, a `GatingMechanism` is created that modulates
the `InputPort` of all the layers in a 3-layered feedforward neural network.  Ordinarily, gating modulates the
*MULTIPLICATIVE_PARAM* of an InputPort's `function <InputPort.function>`.  In the example, this is changed so that
it *adds* the `value <GatingSignal.value>` of the `GatingSignal` to the `value <InputPort.value>` of each InputPort::

    >>> my_input_layer = pnl.TransferMechanism(size=3)
    >>> my_hidden_layer = pnl.TransferMechanism(size=5)
    >>> my_output_layer = pnl.TransferMechanism(size=2)
    >>> my_gating_mechanism = pnl.GatingMechanism(gating_signals=[{pnl.NAME: 'GATE_ALL',
    ...                                                            pnl.PROJECTIONS: [my_input_layer,
    ...                                                                              my_hidden_layer,
    ...                                                                              my_output_layer]}],
    ...                                           modulation=pnl.ADDITIVE)

Note that, again, the **gate** are listed as Mechanisms, since in this case it is their primary InputPorts that are
to be gated. Since they are all listed in a single entry of a `specification dictionary <GatingSignal_Specification>`,
they will all be gated by a single GatingSignal named ``GATE_ALL``, that will send a `GatingProjection` to the
InputPort of each of the Mechanisms listed (the next example shows how different InputPorts can be differentially
gated by a `GatingMechanism`). Finally, note that the value of the **modulation** arguent specified for the
`GatingMechanism` (and therefore the default for its GatingSignals) pertains to the `function <InputPort.function>`
of each `InputPort`. By default that is a `Linear` function, the *ADDITIVE_PARAM* of which is its `intercept
<Linear.intercept>` parameter. Therefore, in the example above, each time the InputPorts are updated, the value of
the GatingSignal will be assigned as the `intercept` of each InputPort's `function <InputPort.function>`, thus adding
that amount to the input to the Port before determining its `value <InputPort.value>`.

**Gate InputPorts differentially**.  In the example above, the InputPorts for all of the Mechanisms were gated
using a single GatingSignal.  In the example below, a different GatingSignal is assigned to the InputPort of each
Mechanism::

    >>> my_gating_mechanism = pnl.GatingMechanism(gating_signals=[{pnl.NAME: 'GATING_SIGNAL_A',
    ...                                                            pnl.MODULATION: pnl.ADDITIVE,
    ...                                                            pnl.PROJECTIONS: my_input_layer},
    ...                                                           {pnl.NAME: 'GATING_SIGNAL_B',
    ...                                                            pnl.PROJECTIONS: [my_hidden_layer,
    ...                                                                              my_output_layer]}])

Here, two GatingSignals are specified as `specification dictionaries <GatingSignal_Specification>`, each of which
contains an entry for the name of the GatingSignal, and a *PROJECTIONS* entry that specifies the Ports to which the
GatingSignal should project (i.e., the ones to be gated).  Once again, the specifications exploit the fact that the
default is to gate the `primary InputPort <InputPort_Primary>` of a Mechanism, so those are what are referenced. The
first dict also contains a  *MODULATION* entry that specifies the value of the `modulation <GatingSignal.modulation>`
attribute for the GatingSignal.  The second one does not, so the default will be used (which, for a GatingSignal, is
*MULTIPLICATIVE*).  Thus, the InputPort of ``my_input_layer`` will be additively modulated by ``GATING_SIGNAL_A``, while
the InputPorts of ``my_hidden_layer`` and ``my_output_layer`` will be multiplicativelymodulated by ``GATING_SIGNAL_B``.

**Creating and assigning stand-alone GatingSignals**.  GatingSignals can also be created on their own, and then later
assigned to a GatingMechanism.  In the example below, the same GatingSignals specified in the previous example are
created directly and then assigned to ``my_gating_mechanism``::

    >>> my_gating_signal_A = pnl.GatingSignal(name='GATING_SIGNAL_A',
    ...                                       modulation=pnl.ADDITIVE,
    ...                                       projections=my_input_layer)
    >>> my_gating_signal_B = pnl.GatingSignal(name='GATING_SIGNAL_B',
    ...                                       projections=[my_hidden_layer,
    ...                                                    my_output_layer])
    >>> my_gating_mechanism = pnl.GatingMechanism(gating_signals=[my_gating_signal_A,
    ...                                                           my_gating_signal_B])


.. _GatingSignal_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.outputport import PRIMARY, SEQUENTIAL, _output_port_variable_getter
from psyneulink.core.components.ports.port import Port_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultGatingAllocation
from psyneulink.core.globals.keywords import \
    CONTEXT, GATE, GATING_PROJECTION, GATING_SIGNAL, INPUT_PORT, INPUT_PORTS, \
    OUTPUT_PORT, OUTPUT_PORTS, OUTPUT_PORT_PARAMS, PROJECTIONS, PROJECTION_TYPE, RECEIVER, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'gating_signal_keywords', 'GatingSignal', 'GatingSignalError',
]


class GatingSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

gating_signal_keywords = {GATE}
# gating_signal_keywords.update(modulatory_signal_keywords)


class GatingSignal(ControlSignal):
    """
    GatingSignal(                                   \
        default_allocation=defaultGatingAllocation, \
        function=Linear())

    A subclass of `ModulatorySignal <ModulatorySignal>` used by a `GatingMechanism` to modulate the value(s)
    of one more `InputPort(s) <InputPort>` and/or `OutputPort(s) <OutputPort>`. See `ControlSignal
    <ControlSignal_Class_Reference>` for additional arguments and attributes).

    COMMENT:
    PortRegistry
    -------------
        All OutputPorts are registered in PortRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    default_allocation : scalar, list or np.ndarray : defaultGatingAllocation
        specifies the template and default value used for `allocation <GatingSignal.allocation>`.

    function : Function or method : default Linear
        specifies the function used to determine the value of the GatingSignal from the value of its
        `owner <GatingMechanism.owner>`.

    Attributes
    ----------

    allocation : float : default: defaultGatingAllocation
        value assigned by the GatingSignal's `owner <Port.owner>`, and used as the `variable <Projection_Base.variable>`
        of the GatingSignal's `function <GatingSignal.function>` to determine its`GatingSignal.intensity`.
    COMMENT:
    FOR DEVELOPERS:  Implemented as an alias of the GatingSignal's variable Parameter
    COMMENT

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        provides the GatingSignal's `value <GatingMechanism.value>`; the default is an identity function that
        passes the input to the GatingMechanism as value for the GatingSignal.

    value : number, list or np.ndarray
        result of the GatingSignal's `function <GatingSignal.function>`
        (same as its `gating_signal <GatingSignal.gating_signal>`).

    intensity : float
        result of the GatingSignal's `function <GatingSignal.function>`;
        assigned as the value of the GatingSignal's GatingProjection, and used to modify the `value <Port_Base.value>`
        of the Port(s) to which the GatingSignal's `GatingProjection(s) <GatingProjection>` project; same as
        `gating_signal <GatingSignal.gating_signal>`.

    gating_signal : number, list or np.ndarray
        result of the GatingSignal's `function <GatingSignal.function>` (same as its `value <GatingSignal.value>`).

    modulation : str
        determines the way in the which `value <GatingSignal.value>` of the GatingSignal is used to modify the `value
        <Port_Base.value>` of the InputPort(s) and/or OutputPort(s) to which the GatingSignal's `GatingProjection(s)
        <GatingProjection>` project.

    efferents : [List[GatingProjection]]
        a list of the `GatingProjections <GatingProjection>` assigned to (i.e., that project from) the GatingSignal.

    """

    #region CLASS ATTRIBUTES

    componentType = GATING_SIGNAL
    componentName = 'GatingSignal'
    paramsType = OUTPUT_PORT_PARAMS

    portAttributes = ControlSignal.portAttributes | {GATE}

    connectsWith = [INPUT_PORT, OUTPUT_PORT]
    connectsWithAttribute = [INPUT_PORTS, OUTPUT_PORTS]
    projectionSocket = RECEIVER
    modulators = []
    projection_type = GATING_PROJECTION

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'OutputPortCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    class Parameters(ControlSignal.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <GatingSignal.variable>`

                    :default value: numpy.array([0.5])
                    :type: ``numpy.ndarray``

                value
                    see `value <GatingSignal.value>`

                    :default value: numpy.array([0.5])
                    :type: ``numpy.ndarray``
                    :read only: True

                allocation_samples
                    see `allocation_samples <GatingSignal.allocation_samples>`

                    :default value: None
                    :type:

                modulation
                    see `modulation <GatingSignal_Modulation>`

                    :default value: None
                    :type:
        """
        variable = Parameter(np.array([defaultGatingAllocation]),
                             aliases='allocation',
                             getter=_output_port_variable_getter,
                             pnl_internal=True, constructor_argument='default_variable'
        )
        value = Parameter(np.array([defaultGatingAllocation]), read_only=True, aliases=['intensity'], pnl_internal=True)
        allocation_samples = Parameter(None, modulable=True)
        modulation = None

        # # Override ControlSignal cost-related attributes and functions
        # cost_options = None
        # intensity_cost = None
        # adjustment_cost = None
        # duration_cost = None
        # cost = None
        # intensity_cost_function = None
        # adjustment_cost_function = None
        # duration_cost_function = None
        # combine_costs_function = None

    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 default_allocation=defaultGatingAllocation,
                 size=None,
                 transfer_function=None,
                 modulation:tc.optional(str)=None,
                 modulates=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.output_ports here (and removing from GatingProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputPorts in GatingProjection._instantiate_sender),

        # Validate sender (as variable) and params
        super().__init__(owner=owner,
                         reference_value=reference_value,
                         default_allocation=default_allocation,
                         size=size,
                         modulation=modulation,
                         modulates=modulates,
                         params=params,
                         name=name,
                         prefs=prefs,
                         transfer_function=transfer_function,
                         **kwargs)

    def _parse_port_specific_specs(self, owner, port_dict, port_specific_spec):
            """Get connections specified in a ParameterPort specification tuple

            Tuple specification can be:
                (Port name, Mechanism)
            [TBI:] (Mechanism, Port name, weight, exponent, projection_specs)

            Returns params dict with CONNECTIONS entries if any of these was specified.

            """
            from psyneulink.core.components.projections.projection import _parse_connection_specs

            params_dict = {}
            port_spec = port_specific_spec

            if isinstance(port_specific_spec, dict):
                return None, port_specific_spec

            elif isinstance(port_specific_spec, tuple):
                port_spec = None
                params_dict[PROJECTIONS] = _parse_connection_specs(connectee_port_type=self,
                                                                   owner=owner,
                                                                   connections=port_specific_spec)
            elif port_specific_spec is not None:
                raise GatingSignalError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                      format(self.__class__.__name__, port_specific_spec))

            if params_dict[PROJECTIONS] is None:
                raise GatingSignalError("PROGRAM ERROR: No entry found in {} params dict for {} "
                                         "with specification of {}, {} or GatingProjection(s) to it".
                                            format(GATING_SIGNAL, INPUT_PORT, OUTPUT_PORT, owner.name))
            return port_spec, params_dict

    def _instantiate_cost_functions(self, context):
        """Override ControlSignal as GatingSignal has not cost functions"""
        pass

    def _get_primary_state(self, mechanism):
        return mechanism.input_port

    @property
    def gating_signal(self):
        return self.value
