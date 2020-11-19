# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  ParameterPort ******************************************************

"""

Contents
--------

  * `ParameterPort_Overview`
  * `ParameterPort_Creation`
      - `ParameterPort_Specification`
  * `ParameterPort_Structure`
  * `ParameterPort_Execution`
  * `ParameterPort_Class_Reference`

.. _ParameterPort_Overview:

Overview
--------

ParameterPorts belong to either a `Mechanism <Mechanism>` or a `Projection <Projection>`. A ParameterPort is created
to represent each `modulatable parameter <ParameterPort_Modulable_Parameters>` of the `Mechanism
<Mechanism>` or a `Projection <Projection>`, as well as those of the component's `function <Component_Function>` and
any of its secondary functions (e.g. `TransferMechanism.integrator_function`). A
ParameterPort provides the current value of the parameter it represents during any relevant computations, and serves as
an interface for parameter modulation.

A ParameterPort can receive one or more `ControlProjections  <ControlProjection>` and/or `LearningProjections
<LearningProjection>` that modify the value returned by the ParameterPort according to the ParameterPort's
`function <ParameterPort.function>`. The Projections received by a ParameterPort  are listed in its `mod_afferents
<ParameterPort.mod_afferents>` attribute.

When the Mechanism or Projection to which a ParameterPort belongs executes, that component and its function use the
ParameterPort's value -- not the parameter attribute's value -- for any computation. A ParameterPort's corresponding
attribute on the Mechanism, Projection, or Function to which it belongs (i.e. MyTransferMech.function.gain),
stores the "base value" of that parameter. The base value of a parameter is the variable of the ParameterPort's
function. The base value can be viewed or changed at any time through this attribute.

The ParameterPort value is available on the ParameterPort itself, as well as the mod_name attribute of the Mechanism
or Projection to which it belongs (i.e. MyTransferMech.mod_gain would return the value of the "gain" ParameterPort
of the MyTransferMech mechanism.)

.. note::
    Either of these options for looking up the value of the ParameterPort will return the ParameterPort value that
    was used during the most recent execution. This means that if the value of MyTransferMech.function.gain (the
    base value) is updated after execution #1, the base value will change immediately, but the ParameterPort value (and
    MyTransferMech.mod_gain) will not be computed again until execution #2.

    As a result, if either MyTransferMech.mod_gain or MyTransferMech.parameter_ports["gain"].value is viewed in between
    execution #1 and execution #2, it will return the gain ParameterPort value that was used during execution 1.

.. _ParameterPort_Creation:

Creating a ParameterPort
-------------------------

ParameterPorts are created automatically when the `Mechanism <Mechanism>` or `Projection <Projection>` to which they
belong is created.  The `owner <Port.owner>` of a ParameterPort must be a `Mechanism <Mechanism>` or `MappingProjection`
(the initialization of a ParameterPort cannot be `deferred <Port_Deferred_Initialization>`). One ParameterPort is
created for each modulable Parameter of its owner, as well as for each modulable Parameter of the owner's
`function <Component.function>` or secondary functions (modulable
Parameters of a Component
are listed in its `Parameters` class, and have the attribute
`modulable <Parameter.modulable>` set to True.)
Each ParameterPort is created using the value specified for the corresponding parameter, as described below.  The
ParameterPorts for the parameters of a Mechanism or Projection are listed in its `parameter_ports` attribute.

COMMENT:
    FOR DEVELOPERS: The instantiation of ParameterPorts for all of the `user_params` of a Component can be
                    suppressed if a *PARAMETER_PORTS* entry is included and set to `NotImplemented` in the
                    default of its class Parameters;  the instantiation of a ParameterPort
                    for an individual parameter in user_params can be suppressed by including it in
                    exclude_from_parameter_ports for the class (or one of its parent classes)
                    (see LearningProjection and EVCControlMechanism for examples, and `note
                    <ParameterPorts_Suppression>` below for additional information about how
                    to suppress creation of a ParameterPort for individual parameters.  This should be done
                    for any parameter than can take a value or a string that is a keyword as its specification
                    (i.e., of the arg for the parameter in the Component's constructor) but should not have a
                    ParameterPort (e.g., input_port and output_port), as otherwise the
                    specification will be interpreted as a numeric parameter (in the case of a value) or
                    a parameter of the keyword's type, a ParameterPort will be created, and then it's value,
                    rather than the parameter's actual value, will be returned when the parameter is accessed
                    using "dot notation" (this is because the getter for an attribute's property first checks
                    to see if there is a ParameterPort for that attribute and, if so, returns the value of the
                    ParameterPort).
COMMENT

.. _ParameterPort_Specification:

*Specifying Parameters*
~~~~~~~~~~~~~~~~~~~~~~~

Parameters can be specified in one of several places:

    * In the **argument** of the constructor for the `Component <Component>` to which the parameter belongs
      (see `Component_Structural_Attributes` for additional details).
    ..
    * In a *parameter specification dictionary* assigned to the **params** argument in the constructor for the
      Component to which the parameter belongs, or any place else the value of a parameter can be specified.
      The entry for each parameter must use the name of the parameter (or a corresponding keyword) as its key,
      and the parameter's specification as its value (see `examples <ParameterPort_Specification_Examples>` below).
      Parameters for a Component's `function <Component.function>` can be specified in an entry with the key
      *FUNCTION_PARAMS*, and a value that is itself a parameter specification dictionary containing an entry for
      each of the function's parameters to be specified.  When a value is assigned to a parameter in a specification
      dictionary, it overrides any value assigned to the argument for the parameter in the Component's constructor.
    ..
    * By direct assignment to the Component's attribute for the parameter
      (see `below <ParameterPort_Modulable_Parameters>`).
    ..
    ..
    * In the **runtime_params** argument of a call to a Composition's `Run` method

.. _ParameterPort_Value_Specification:

The specification of the initial value of a parameter can take any of the following forms:

    .. _ParameterPort_Value_Assignment:

    * **Value** -- this must be a valid value for the parameter. It creates a default ParameterPort,
      assigns the parameter's default value as the ParameterPort's `value <ParameterPort.value>`,
      and assigns the parameter's name as the name of the ParameterPort.

    * **ParameterPort reference** -- this must refer to an existing **ParameterPort** object; its name must be the
      name of a parameter of the owner or of the owner's `function <Component.function>`, and its value must be a valid
      one for the parameter.

      .. note::
          This capability is provided for generality and potential
          future use, but its current use is not advised.

    .. _ParameterPort_Modulatory_Specification:

    * **Modulatory specification** -- this can be an existing `ControlSignal` or `ControlProjection`,
      a `LearningSignal` or `LearningProjection`, a constructor or the class name for any of these, or the
      keywords *CONTROL*, *CONTROL_PROJECTION*, *LEARNING*, or *LEARNING_PROJECTION*.  Any of these create a default
      ParameterPort, assign the parameter's default value as the ParameterPort's `value <ParameterPort.value>`,
      and assign the parameter's name as the name of the ParameterPort.  They also create and/or assign the
      corresponding ModulatorySignal and ModulatoryProjection, and assign the ParameterPort as the
      ModulatoryProjection's `receiver <Projection_Base.receiver>`. If the ModulatorySignal and/or
      ModulatoryProjection already exist, their value(s) must be valid one(s) for the parameter.  Note that only
      Control and Learning Modulatory components can be assigned to a ParameterPort (Gating components cannot be
      used -- they can only be assigned to `InputPorts <InputPort>` and `OutputPorts <OutputPort>`).

    .. _ParameterPort_Tuple_Specification:

    * **2-item tuple:** *(<value>, <Modulatory specification>)* -- this creates a default ParameterPort, uses the value
      specification (1st item) as parameter's `value assignment <ParameterPort_Value_Assignment>`, and assigns the
      parameter's name as the name of the ParameterPort.  The Modulatory specification (2nd item) is used as the
      ParameterPort's `modulatory assignment <ParameterPort_Modulatory_Specification>`, and the ParameterPort
      is assigned as the `receiver <Projection_Base.receiver>` for the corresponding `ModulatoryProjection
      <ModulatoryProjection>`.

      .. note::
          Currently, the `function <Component.function>` of a Component, although it can be specified as a parameter
          value, cannot be assigned a `ModulatorySignal <ModulatorySignal>` or modified in the **runtime_params**
          argument of a call to a Mechanism's `execute <Mechanism_Base.execute>` method. This may change in the future.

The value specified for a parameter (either explicitly or by default) is assigned to an attribute of the Component or
of the Component's `function <Mechanism_Base.function>` to which the parameter belongs.  The attribute has the same
name as the parameter, and can be referenced using standard Python attribute ("dot") notation;  for example, the value
of a parameter named *param* is assigned to an attribute named ``param`` that can be referenced as
``my_component.param``). The parameter's value is assigned as the **default value** for the ParameterPort.

.. _ParameterPorts_Suppression:

.. note::
   If the value of a parameter is specified as `NotImplemented`, or any non-numeric value that is not one of those
   listed above, then no ParameterPort is created and the parameter cannot be modified by a `ModulatorySignal
   <ModulatorySignal>` or in the **runtime_params** argument of a call to a Mechanism's `execute
   <Mechanism_Base.execute>` method.

.. _ParameterPort_Specification_Examples:

*Examples*
~~~~~~~~~~

In the following example, a Mechanism is created by specifying two of its parameters, as well as its
`function <Component.function>` and two of that function's parameters, each using a different specification format::

    >>> import psyneulink as pnl
    >>> my_mechanism = pnl.RecurrentTransferMechanism(
    ...                         size=5,
    ...                         noise=pnl.ControlSignal(),
    ...                         function=pnl.Logistic(
    ...                                         gain=(0.5, pnl.ControlSignal),
    ...                                         bias=(1.0, pnl.ControlSignal(modulation=pnl.ADDITIVE))))

COMMENT:
    If assigning a default ControlSignal makes the noise value the same as the
    default noise value, why are we using a ControlSignal here??
COMMENT

The first argument of the constructor for the Mechanism specifies its `size <Component.size>` parameter by
directly assigning a value to it.  The second specifies the `noise <RecurrentTransferMechanism.noise>` parameter
by assigning a default `ControlSignal`;  this will use the default value of the
`noise <RecurrentTransferMechanism.noise>` attribute.  The **function** argument is specified using the constructor for
a `Logistic` function, that specifies two of its parameters.  The `gain <Logistic.gain>` parameter
is specified using a tuple, the first item of which is the value to be assigned, and the second specifies
a default `ControlSignal`.  The `bias <Logistic.bias>` parameter is also specified using a tuple,
in this case with a constructor for the ControlSignal that specifies its `modulation <ModulatorySignal.modulation>`
parameter.

In the following example, a `MappingProjection` is created, and its
`matrix <MappingProjection.MappingProjection.matrix>` parameter is assigned a random weight matrix (using a
`matrix keyword <Matrix_Keywords>`) and `LearningSignal`::

    >>> my_input_mechanism = pnl.TransferMechanism()
    >>> my_output_mechanism = pnl.TransferMechanism()
    >>> my_mapping_projection = pnl.MappingProjection(sender=my_input_mechanism,
    ...                                               receiver=my_output_mechanism,
    ...                                               matrix=(pnl.RANDOM_CONNECTIVITY_MATRIX,
    ...                                                       pnl.LearningSignal))

.. note::
   The `matrix <MappingProjection.MappingProjection.matrix>` parameter belongs to the MappingProjection's `function
   <Projection_Base.function>`;  however, since it has only one standard function, its arguments are available in the
   constructor for the Projection (see `here <User_Modifiable_Parameters>` for a more detailed explanation).

The example below shows how to specify the parameters in the first example using a parameter specification dictionary::

    >>> my_mechanism = pnl.RecurrentTransferMechanism(
    ...                      noise=5,
    ...                      params={pnl.NOISE: pnl.CONTROL,
    ...                              pnl.FUNCTION: pnl.Logistic,
    ...                              pnl.FUNCTION_PARAMS:{
    ...                                     pnl.GAIN:(0.5,pnl.ControlSignal),
    ...                                     pnl.BIAS:(1.0,pnl.ControlSignal(modulation=pnl.ADDITIVE))}})

There are several things to note here.

First, the parameter specification dictionary must be assigned to the **params** argument of the constructor. Note that
if the parameter is specified in a parameter specification dictionary, the key for the parameter must be a string that
is the same as the name of parameter (i.e., identical to how it appears as an arg in the constructor; as is shown
for **noise** in the example), or using a keyword that resolves to such a string (as shown for *NOISE* in the
example).

Second, both methods for specifying a parameter -- directly in an argument for the parameter, or in an entry of a
parameter specification dictionary -- can be used within the same constructor.

If a particular parameter is specified in both ways (as is the case for **noise** in the example), the value in the
parameter specification dictionary takes priority (i.e., it is the value that will be assigned to the parameter).

Finally, the keyword *FUNCTION_PARAMS* can be used in a parameter specification dictionary to specify
parameters of the Component's `function <Component.function>`, as shown for the **gain** and **bias** parameters of
the Logistic function in the example.

The example below shows how to access ParameterPort values vs base values, and demonstrates their differences:

    >>> my_transfer_mechanism = pnl.TransferMechanism(
    ...                      noise=5.0,
    ...                      function=pnl.Linear(slope=2.0))
    >>> assert my_transfer_mechanism.noise.base == 5.0
    >>> assert my_transfer_mechanism.mod_noise == [5.0]
    >>> assert my_transfer_mechanism.function.slope.base == 2.0
    >>> assert my_transfer_mechanism.mod_slope == [2.0]

Notice that the noise attribute, which stores the base value for the noise ParameterPort of my_transfer_mechanism, is
on my_transfer_mechanism, while the slope attribute, which stores the base value for the slope ParameterPort of
my_transfer_mechanism, is on my_transfer_mechanism's function. However, mod_noise and mod_slope are both properties on
my_transfer_mechanism.

    >>> my_transfer_mechanism.noise.base = 4.0
    >>> my_transfer_mechanism.function.slope.base = 1.0
    >>> assert my_transfer_mechanism.noise.base == 4.0
    >>> assert my_transfer_mechanism.mod_noise == [5.0]
    >>> assert my_transfer_mechanism.function.slope.base == 1.0
    >>> assert my_transfer_mechanism.mod_slope == [2.0]

When the base values of noise and slope are updated, we can inspect these attributes immediately and observe that they
have changed. We do not observe a change in mod_noise or mod_slope because the ParameterPort value will not update
until the mechanism executes.

    >>> my_transfer_mechanism.execute([10.0])
    array([[14.]])
    >>> assert my_transfer_mechanism.noise.base == 4.0
    >>> assert my_transfer_mechanism.mod_noise == [4.0]
    >>> assert my_transfer_mechanism.function.slope.base == 1.0
    >>> assert my_transfer_mechanism.mod_slope == 1.0

Now that the mechanism has executed, we can see that each ParameterPort evaluated its function with the base value,
producing a modulated noise value of 4.0 and a modulated slope value of 1.0. These values were used by
my_transfer_mechanism and its Linear function when the mechanism executed.

.. _ParameterPort_Structure:

Structure
---------

Every ParameterPort is owned by a `Mechanism <Mechanism>` or `MappingProjection`. It can receive one or more
`ControlProjections <ControlProjection>` or `LearningProjections <LearningProjection>`, that are listed in its
`mod_afferents <ParameterPort.mod_afferents>` attribute.  A ParameterPort cannot receive
`PathwayProjections <PathwayProjection>` or `GatingProjections <GatingProjection>`.  When the ParameterPort is
updated (i.e., its owner is executed), it uses the values of its ControlProjections and LearningProjections to
determine whether and how to modify its parameter's attribute value, which is then assigned as the ParameterPort's
`value <ParameterPort.value>` (see `ParameterPort_Execution` for addition details). ParameterPorts have the
following core attributes:

* `variable <ParameterPort.variable>` - the parameter's attribute value; that is, the value assigned to the
  attribute for the parameter of the ParameterPort's owner;  it can be thought of as the parameter's "base" value.
  It is used by its `function <ParameterPort.function>` to determine the ParameterPort's
  `value <ParameterPort.value>`.  It must match the format (the number and type of elements) of the parameter's
  attribute value.

* `mod_afferents <ParameterPort.mod_afferents>` - lists the `ModulatoryProjections <ModulationProjection>` received
  by the ParameterPort.  These specify either modify the ParameterPort's `function <ParameterPort.function>`, or
  directly assign the `value <ParameterPort.value>` of the ParameterPort itself (see `ModulatorySignals_Modulation`).

* `function <ParameterPort.function>` - takes the parameter's attribute value as its input, modifies it under the
  influence of any `ModulatoryProjections` it receives (listed in `mod_afferents <ParameterPort.mod_afferents>`,
  and assigns the result as the ParameterPort's `value <ParameterPort.value>` which is used as the parameter's
  "actual" value.

* `value <ParameterPort.value>` - the result of `function <ParameterPort.function>`; used by the ParameterPort's
  owner as the value of the parameter for which the ParameterPort is responsible.

.. _ParameterPort_Modulable_Parameters:

All of the modulable parameters of a Component -- that is, for which it has ParameterPorts --
are listed in its `Parameters` class, and have the attribute
`modulable <Parameter.modulable>` set to True.  The
ParameterPorts for a Mechanism or Projection are listed in its :keyword:`parameter_ports` attribute, which is also
read-only.

An initial value can be assigned to a parameter in the corresponding argument of the constructor for the Component
(see `above <ParameterPort_Value_Specification>`.  Parameter values can also be modified by a assigning a value to
the corresponding attribute.
The parameters of a Component's function can be modified by assigning a value to the corresponding attribute of the
Component's `function <Component.function>` attribute (e.g., ``myMechanism.function.my_parameter``).
See `Mechanism_ParameterPorts` for additional information.


.. _ParameterPort_Execution:

Execution
---------

A ParameterPort cannot be executed directly.  It is executed when the Component to which it belongs is executed.
When this occurs, the ParameterPort executes any `ModulatoryProjections` it receives, the values of which
modulate parameters of the ParameterPort's `function <ParameterPort.function>`.  The ParameterPort then calls
its `function <ParameterPort.function>` and the result is assigned as its `value <ParameterPort.value>`.  The
ParameterPort's `value <ParameterPort.value>` is used as the value of the corresponding parameter by the Component,
or by its own `function <Component.function>`.

.. note::
   It is important to note the distinction between the `function <ParameterPort.function>` of a ParameterPort,
   and the `function <Component.function>` of the Component to which it belongs. The former is used to determine the
   value of a parameter used by the latter (see `figure <ModulatorySignal_Anatomy_Figure>`, and `Port_Execution` for
   additional details).

.. _ParameterPort_Class_Reference:

Class Reference
---------------

"""

import collections
from copy import deepcopy
import inspect
import operator
import types
import warnings

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import Component, parameter_keywords
from psyneulink.core.components.functions.function import get_param_value_for_keyword
from psyneulink.core.components.shellclasses import Mechanism, Projection, Function
from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import ModulatorySignal
from psyneulink.core.components.ports.port import PortError, Port_Base, _instantiate_port, port_type_keywords
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    CONTEXT, CONTROL_PROJECTION, CONTROL_SIGNAL, CONTROL_SIGNALS, FUNCTION, FUNCTION_PARAMS, \
    LEARNING_SIGNAL, LEARNING_SIGNALS, MECHANISM, NAME, PARAMETER_PORT, PARAMETER_PORTS, \
    PARAMETER_PORT_PARAMS, PATHWAY_PROJECTION, PROJECTION, PROJECTIONS, PROJECTION_TYPE, REFERENCE_VALUE, SENDER, VALUE
from psyneulink.core.globals.parameters import ParameterBase, ParameterAlias, SharedParameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities \
    import ContentAddressableList, ReadOnlyOrderedDict, is_iterable, is_numeric, is_value_spec, iscompatible, is_instance_or_subclass, UtilitiesError, gen_friendly_comma_str

__all__ = [
    'ParameterPort', 'ParameterPortError', 'port_type_keywords',
]

port_type_keywords = port_type_keywords.update({PARAMETER_PORT})


class ParameterPortList(ContentAddressableList):

    separator = '-'
    legal_key_type_strings = ContentAddressableList.legal_key_type_strings + ['Parameter']

    def __init__(
        self,
        component_type,
        key=None,
        list=None,
        name=None,
        owner=None,
        **kwargs
    ):
        # cache, Parameter keys added when creating Ports, others upon lookup
        self.parameter_mapping = {}
        self.owner = owner

        super().__init__(component_type, key, list, name, **kwargs)

    def __contains__(self, item):
        try:
            return super().__contains__(item)
        except ParameterPortError:
            return False

    def __getitem__(self, key):
        try:
            return self.parameter_mapping[key]
        except KeyError:
            pass

        try:
            return super().__getitem__(key)
        except TypeError as e:
            # ContentAddressableList throws TypeError when key/index lookup fails
            names = self._get_possible_port_names(key)
            possible_ports = set()
            for name in names:
                try:
                    r = super().__getitem__(name)
                    possible_ports.add(r)
                except TypeError:
                    pass
            if len(possible_ports) == 0:
                raise e from None
            elif len(possible_ports) == 1:
                res = next(iter(possible_ports))
            else:
                raise ParameterPortError(
                    f'Multiple ParameterPorts for {key} exist. Did you want'
                    f' {gen_friendly_comma_str(sorted([p.name for p in possible_ports]))}?'
                ) from None
        except UtilitiesError as e:
            # ContentAddressableList throws UtilitiesError if key is not an int
            # or string. handle only Parameter key here
            if not isinstance(key, ParameterBase):
                raise e from None

            try:
                final_source = key.final_source
            except AttributeError:
                final_source = key

            try:
                res = self.parameter_mapping[final_source]
            except KeyError:
                try:
                    raise ParameterPortError(
                        f'No ParameterPort corresponds to {key._owner._owner}'
                        f'.parameters.{key.name}'
                    ) from None
                except AttributeError:
                    raise e from None

        if res is not None:
            self.parameter_mapping[key] = res

        return res

    def _get_possible_port_names(self, param_name):
        """
            Returns:
                a list of possible parameter port names to check if
                *param_name* is actually an alias or alias-with-suffix
                (e.g. "leak" is an alias of "integration_rate", and
                "leak__integrator_function" should refer to
                "integration_rate__integrator_function")
        """
        unsuffixed_name = ParameterPortList._get_base_name(param_name)
        if unsuffixed_name == param_name:
            # all possible function-suffixed names
            names = sorted([
                p.name for p in self.owner.parameters
                if is_instance_or_subclass(p.default_value, Function)
            ])
            # put 'function' at beginning
            try:
                function_index = names.index(FUNCTION)
                names = (
                    [names[function_index]]
                    + names[0:function_index]
                    + names[function_index + 1:]
                )
            except ValueError:
                pass

            names = [self._get_explicit_name(param_name, name) for name in names]
        else:
            names = []

        # try to get a Parameter that corresponds to param_name, which
        # can have a "shared parameter suffix" that disambiguates which
        # desired port it refers to if there are multiple
        try:
            param = getattr(self.owner.parameters, param_name)
        except AttributeError:
            try:
                param = getattr(self.owner.parameters, unsuffixed_name)
            except AttributeError:
                return names

        # if it's a shared parameter with identical name, there are no
        # other aliases we need to add
        try:
            source_name = param.source.name
        except AttributeError:
            return names

        if source_name != param.name:
            if unsuffixed_name == param_name:
                # basic alias, e.g. "leak" -> "integration_rate"
                names.append(source_name)
            else:
                # alias with suffix, e.g. "leak__function"
                # -> "integration_rate__function"
                suffix = ParameterPortList._get_suffix(param_name)
                names.append(
                    ParameterPortList._get_explicit_name(source_name, suffix)
                )

            if isinstance(param, ParameterAlias):
                # alias to another alias or a shared parameter
                # e.g. leak -> integration_rate -> rate
                names.extend(self._get_possible_port_names(source_name))
            else:
                # e.g. integration_rate__integrator_function
                # -> rate__integrator_function
                names.append(
                    ParameterPortList._get_explicit_name(
                        source_name,
                        param.attribute_name
                    )
                )

        return names

    @classmethod
    def _get_explicit_name(cls, port_name, parameter_name=None):
        return f'{port_name}{cls.separator}{parameter_name}'

    @classmethod
    def _get_base_name(cls, explicit_name):
        try:
            return explicit_name.split(cls.separator)[0]
        except IndexError:
            return explicit_name

    @classmethod
    def _get_suffix(cls, explicit_name):
        try:
            return explicit_name.split(cls.separator)[1]
        except IndexError:
            return ''


class ParameterPortError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ParameterPort(Port_Base):
    """
    ParameterPort(                                           \
        owner,                                               \
        reference_value=None                                 \
        function=LinearCombination(operation=PRODUCT),       \


    Subclass of `Port <Port>` that represents and possibly modifies the parameter of a `Mechanism <Mechanism>`,
    `Projection <Projection>`, or its `Function`. See `Port_Class_Reference` for additional arguments and attributes.

    COMMENT:
    PortRegistry
    -------------
        All ParameterPorts are registered in PortRegistry, which maintains an entry for the subclass,
        a count for all instances of it, and a dictionary of those instances

    COMMENT

    Arguments
    ---------

    owner : Mechanism or MappingProjection
        the `Mechanism <Mechanism>` or `MappingProjection` to which to which the ParameterPort belongs; it must be
        specified or determinable from the context in which the ParameterPort is created (the initialization of a
        ParameterPort cannot be `deferred <Port_Deferred_Initialization>`. The owner of a ParameterPort
        for the parameter of a `function <Component.function>` should be specified as the Mechanism or Projection to
        which the function belongs.

    reference_value : number, list or np.ndarray
        specifies the default value of the parameter for which the ParameterPort is responsible.

    variable : number, list or np.ndarray
        specifies the parameter's initial value and attribute value — that is, the value of the attribute of the
        ParameterPort's owner or its `function <Component.function>` assigned to the parameter.

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to convert the parameter's attribute value (same as the ParameterPort's
        `variable <ParameterPort.variable>`) to the ParameterPort's `value <ParameterPort.value>`.


    Attributes
    ----------

    mod_afferents : Optional[List[Projection]]
        a list of the `ModulatoryProjection <ModulatoryProjection>` that project to the ParameterPort (i.e.,
        for which it is a `receiver <Projection_Base.receiver>`); these can be `ControlProjection(s)
        <ControlProjection>` and/or `LearningProjection(s) <LearningProjection>`, but not `GatingProjection
        <GatingProjection>`.  The `value <ModulatoryProjection_Base.value>` of each must match the format
        (number and types of elements) of the ParameterPort's `variable <ParameterPort.variable>`.

    variable : number, list or np.ndarray
        the parameter's attribute value — that is, the value of the attribute of the
        ParameterPort's owner or its `function <Component.function>` assigned to the parameter.

    function : Function : default Linear
        converts the parameter's attribute value (same as the ParameterPort's `variable <ParameterPort.variable>`)
        to the ParameterPort's `value <ParameterPort.value>`, under the influence of any
        `ModulatoryProjections <ModulatoryProjection>` received by the ParameterPort (and listed in its
        `mod_afferents <ParameterPort.mod_afferents>` attribute.  The result is assigned as the ParameterPort's
        `value <ParameterPort>`.

    value : number, List[number] or np.ndarray
        the result returned by the ParameterPort's `function <ParameterPort.function>`, and used by the
        ParameterPort's owner or its `function <Component.function>` as the value of the parameter for which the
        ParmeterPort is responsible.  Note that this is not necessarily the same as the parameter's attribute value
        (that is, the value of the owner's attribute for the parameter), since the ParameterPort's
        `function <ParameterPort.function>` may modify the latter under the influence of its
        `mod_afferents <ParameterPort.mod_afferents>`.

    """

    #region CLASS ATTRIBUTES

    componentType = PARAMETER_PORT
    paramsType = PARAMETER_PORT_PARAMS

    portAttributes = Port_Base.portAttributes

    connectsWith = [CONTROL_SIGNAL, LEARNING_SIGNAL]
    connectsWithAttribute = [CONTROL_SIGNALS, LEARNING_SIGNALS]
    projectionSocket = SENDER
    modulators = [CONTROL_SIGNAL, LEARNING_SIGNAL]
    canReceive = modulators
    projection_type = CONTROL_PROJECTION

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ParameterPortCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    #endregion

    tc.typecheck
    def __init__(self,
                 owner,
                 reference_value=None,
                 variable=None,
                 size=None,
                 function=None,
                 projections=None,
                 params=None,
                 name=None,
                 parameter_name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        # If context is not COMPONENT or CONSTRUCTOR, raise exception
        context = kwargs.pop(CONTEXT, None)
        if context is None:
            raise ParameterPortError(f"Contructor for {self.__class__.__name__} cannot be called directly"
                                      f"(context: {context}")

        # FIX: UPDATED TO INCLUDE LEARNING [CHANGE THIS TO INTEGRATOR FUNCTION??]
        # # Reassign default for MATRIX param of MappingProjection
        # if isinstance(owner, MappingProjection) and name is MATRIX:
        #     function = LinearCombination(operation=SUM)

        self.reference_value = reference_value

        # Validate sender (as variable) and params
        # Note: pass name of Mechanism (to override assignment of componentName in super.__init__)
        super(ParameterPort, self).__init__(owner,
                                            variable=variable,
                                            size=size,
                                            projections=projections,
                                            function=function,
                                            params=params,
                                            name=name,
                                            prefs=prefs,
                                            context=context)

    def _validate_against_reference_value(self, reference_value):
        """Validate that value of the Port is compatible with the reference_value

        reference_value is the value of the parameter to which the ParameterPort is assigned
        """
        if reference_value is not None and not iscompatible(np.squeeze(reference_value), np.squeeze(self.defaults.value)):
            iscompatible(np.squeeze(reference_value), np.squeeze(self.defaults.value))
            name = self.name or ""
            raise ParameterPortError("Value specified for {} {} of {} ({}) is not compatible "
                                      "with its expected format ({})".
                                      format(name, self.componentName, self.owner.name, self.defaults.value, reference_value))

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of Port's constructor

        Disallow any PathwayProjections
        Call _instantiate_projections_to_port to assign ModulatoryProjections to .mod_afferents

        """

        # MODIFIED 7/8/17
        # FIX:  THIS SHOULD ALSO LOOK FOR OTHER FORMS OF SPECIFICATION
        # FIX:  OF A PathwayProjection (E.G., TARGET PORT OR MECHANISM)

        from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base
        pathway_projections = [proj for proj in projections if isinstance(proj, PathwayProjection_Base)]
        if pathway_projections:
            pathway_proj_names = []
            for proj in pathway_projections:
                pathway_proj_names.append(proj.name + ' ')
            raise PortError("{} not allowed for {}: {}".
                             format(PathwayProjection_Base.__self__.__name__,
                                    self.__class__.__name__,
                                    pathway_proj_names))

        self._instantiate_projections_to_port(projections=projections, context=context)

    def _check_for_duplicate_projections(self, projection):
        """Check if projection is redundant with one in mod_afferents of ParameterPort

        Check for any instantiated projection in mod_afferents with the same sender as projection
        or one in deferred_init status with sender specification that is the same type as projection.

        Returns redundant Projection if found, otherwise False.
        """

        duplicate = next(iter([proj for proj in self.mod_afferents
                               if ((proj.sender == projection.sender and proj != projection)
                                   or (proj.initialization_status == ContextFlags.DEFERRED_INIT
                                       and proj._init_args[SENDER] == type(projection.sender)))]), None)
        if duplicate and self.verbosePref or self.owner.verbosePref:
            from psyneulink.core.components.projections.projection import Projection
            warnings.warn(f'{Projection.__name__} from {projection.sender.name}  {projection.sender.__class__.__name__}'
                          f' of {projection.sender.owner.name} to {self.name} {self.__class__.__name__} of '
                          f'{self.owner.name} already exists; will ignore additional one specified ({projection.name}).')
        return duplicate

    @tc.typecheck
    def _parse_port_specific_specs(self, owner, port_dict, port_specific_spec):
        """Get connections specified in a ParameterPort specification tuple

        Tuple specification can be:
            (port_spec, projections)
        Assumes that port_spec has already been extracted and used by _parse_port_spec

        Returns params dict with PROJECTIONS entries if any of these was specified.

        """
        from psyneulink.core.components.projections.projection import _parse_connection_specs, _is_projection_spec

        params_dict = {}
        port_spec = port_specific_spec

        if isinstance(port_specific_spec, dict):
            return None, port_specific_spec

        elif isinstance(port_specific_spec, tuple):

            tuple_spec = port_specific_spec

            # GET PORT_SPEC (PARAM VALUE) AND ASSIGN PROJECTIONS_SPEC **********************************************

            # 2-item tuple specification
            if len(tuple_spec) == 2:

                # 1st item is a value, so treat as Port spec (and return to _parse_port_spec to be parsed)
                #   and treat 2nd item as Projection specification
                if is_numeric(tuple_spec[0]):
                    port_spec = tuple_spec[0]
                    reference_value = port_dict[REFERENCE_VALUE]
                    # Assign value so sender_dim is skipped below
                    # (actual assignment is made in _parse_port_spec)
                    if reference_value is None:
                        port_dict[REFERENCE_VALUE]=port_spec
                    elif not iscompatible(port_spec, reference_value):
                        raise PortError("Value in first item of 2-item tuple specification for {} of {} ({}) "
                                         "is not compatible with its {} ({})".
                                         format(ParameterPort.__name__, owner.name, port_spec,
                                                REFERENCE_VALUE, reference_value))
                    projections_spec = tuple_spec[1]

                elif _is_projection_spec(tuple_spec[0], include_matrix_spec=True):
                    port_spec, projections_spec = tuple_spec

                # Tuple is Projection specification that is used to specify the Port,
                else:
                    # return None in port_spec to suppress further, recursive parsing of it in _parse_port_spec
                    port_spec = None
                    if tuple_spec[0] != self:
                        # If 1st item is not the current port (self), treat as part of the projection specification
                        projections_spec = tuple_spec
                    else:
                        # Otherwise, just use 2nd item as projection spec
                        port_spec = None
                        projections_spec = tuple_spec[1]

            # 3- or 4-item tuple specification
            elif len(tuple_spec) in {3,4}:
                # Tuple is projection specification that is used to specify the Port,
                #    so return None in port_spec to suppress further, recursive parsing of it in _parse_port_spec
                port_spec = None
                # Reduce to 2-item tuple Projection specification
                projection_item = tuple_spec[3] if len(tuple_spec)==4 else None
                projections_spec = (tuple_spec[0],projection_item)

            # GET PROJECTIONS IF SPECIFIED *************************************************************************

            try:
                projections_spec
            except UnboundLocalError:
                pass
            else:
                try:
                    params_dict[PROJECTIONS] = _parse_connection_specs(self,
                                                                       owner=owner,
                                                                       connections=projections_spec)

                    # Parse the value of all of the Projections to get/validate parameter value
                    from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
                    from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection

                    for projection_spec in params_dict[PROJECTIONS]:
                        if port_dict[REFERENCE_VALUE] is None:
                            # FIX: - PUTTING THIS HERE IS A HACK...
                            # FIX:     MOVE TO _parse_port_spec UNDER PROCESSING OF ProjectionTuple SPEC
                            # FIX:     USING _get_port_for_socket
                            # from psyneulink.core.components.projections.projection import _parse_projection_spec

                            # defaults.value?
                            mod_signal_value = projection_spec.port.value \
                                if isinstance(projection_spec.port, Port_Base) else None

                            mod_projection = projection_spec.projection
                            if isinstance(mod_projection, dict):
                                if mod_projection[PROJECTION_TYPE] not in {ControlProjection, LearningProjection}:
                                    raise ParameterPortError("PROGRAM ERROR: {} other than {} or {} ({}) found "
                                                              "in specification tuple for {} param of {}".
                                                              format(Projection.__name__,
                                                                     ControlProjection.__name__,
                                                                     LearningProjection.__name__,
                                                                     mod_projection, port_dict[NAME], owner.name))
                                elif VALUE in mod_projection:
                                    mod_proj_value = mod_projection[VALUE]
                                else:
                                    mod_proj_value = None
                            elif isinstance(mod_projection, Projection):
                                if not isinstance(mod_projection, (ControlProjection, LearningProjection)):
                                    raise ParameterPortError("PROGRAM ERROR: {} other than {} or {} ({}) found "
                                                              "in specification tuple for {} param of {}".
                                                              format(Projection.__name__,
                                                                     ControlProjection.__name__,
                                                                     LearningProjection.__name__,
                                                                     mod_projection, port_dict[NAME], owner.name))
                                elif mod_projection.initialization_status == ContextFlags.DEFERRED_INIT:
                                    continue
                                mod_proj_value = mod_projection.defaults.value
                            else:
                                raise ParameterPortError("Unrecognized Projection specification for {} of {} ({})".
                                                      format(self.name, owner.name, projection_spec))

                            # FIX: 11/25/17 THIS IS A MESS:  CHECK WHAT IT'S ACTUALLY DOING
                            # If ModulatoryProjection's value is not specified, try to assign one
                            if mod_proj_value is None:
                                # If not specified for Port, assign that
                                if VALUE not in port_dict or port_dict[VALUE] is None:
                                    port_dict[VALUE] = mod_signal_value
                                # If value has been assigned, make sure value is the same for ModulatorySignal
                                elif port_dict[VALUE] != mod_signal_value:
                                    # If the values differ, assign None so that Port's default is used
                                    port_dict[VALUE] = None
                                    # No need to check any more ModulatoryProjections
                                    break

                            #
                            else:
                                port_dict[VALUE] = mod_proj_value

                except ParameterPortError:
                    raise ParameterPortError("Tuple specification in {} specification dictionary "
                                          "for {} ({}) is not a recognized specification for one or more "
                                          "{}s, {}s, or {}s that project to it".
                                          format(ParameterPort.__name__,
                                                 owner.name,
                                                 projections_spec,
                                                 Mechanism.__name__,
                                                 ModulatorySignal.__name__,
                                                 Projection.__name__))

        elif port_specific_spec is not None:
            raise ParameterPortError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, port_specific_spec))

        return port_spec, params_dict

    @staticmethod
    def _get_port_function_value(owner, function, variable):
        """Return parameter variable (since ParameterPort's function never changes the form of its variable"""
        return variable

    def _get_variable_from_projections(self, context=None):
        """
        Get backingfield ("base") value of param of function of Mechanism to which the ParameterPort belongs.
        """

        # FIX 3/6/19: source does not yet seem to have been assigned to owner.function
        return self.source._get(context)

    @property
    def pathway_projections(self):
        raise ParameterPortError("PROGRAM ERROR: Attempt to access {} for {}; {}s do not have {}s".
                                  format(PATHWAY_PROJECTION, self.name, PARAMETER_PORT, PATHWAY_PROJECTION))

    @pathway_projections.setter
    def pathway_projections(self, value):
        raise ParameterPortError("PROGRAM ERROR: Attempt to assign {} to {}; {}s cannot accept {}s".
                                  format(PATHWAY_PROJECTION, self.name, PARAMETER_PORT, PATHWAY_PROJECTION))

def _instantiate_parameter_ports(owner, function=None, context=None):
    """Call _instantiate_parameter_port for all modulable parameters to instantiate ParameterPorts for them

    If owner.parameter_port is None or False:
        - no ParameterPorts will be instantiated.
    Otherwise, instantiate ParameterPort for each modulable parameter
    :param function:

    """

    # TBI / IMPLEMENT: use specs to implement ParameterPorts below

    owner._parameter_ports = ParameterPortList(
        component_type=ParameterPort,
        name=owner.name + '.parameter_ports',
        owner=owner,
    )

    # Check that all ParameterPorts for owner have not been explicitly suppressed
    try:
        if owner.parameter_ports is NotImplemented:
            return
    except KeyError:
        # PARAMETER_PORTS not specified at all, so OK to continue and construct them
        pass

    # Instantiate ParameterPort for each modulable Parameter on
    # function and owner. function is first because in some
    # cases a Parameter will be specified on both, and the function's
    # values/defaults should take precedence
    def skip_parameter_port(parameter):
        return (
            isinstance(parameter, (ParameterAlias, SharedParameter))
            or parameter.name in owner.exclude_from_parameter_ports
            or not parameter.modulable
        )

    port_parameters = collections.defaultdict(set)
    port_aliases = set()

    # function may be a custom function not yet parsed to a UDF
    # function may also be a Function class, in which case parameter
    # ports are still created for the modulable Parameters

    for p in owner.parameters:
        func = p.default_value
        if (
            not p.reference
            and is_instance_or_subclass(func, Function)
            and not isinstance(p, (ParameterAlias, SharedParameter))
        ):
            for func_param in func.parameters:
                if not skip_parameter_port(func_param):
                    port_parameters[func_param.name].add(p.name)
        if isinstance(p, ParameterAlias):
            port_aliases.add(p.name)

    for parameter_port_name in port_parameters:
        if (
            len(port_parameters[parameter_port_name]) > 1
            or parameter_port_name in port_aliases
        ):
            add_suffix = True
        else:
            add_suffix = False

        for corresponding_parameter_component_name in port_parameters[parameter_port_name]:
            corresponding_parameter_component = getattr(
                owner.parameters,
                corresponding_parameter_component_name
            )._get(context)

            p = getattr(
                corresponding_parameter_component.parameters,
                parameter_port_name
            )

            # .function is not finalized yet, because this happens before
            # _instantiate_function
            if corresponding_parameter_component_name is FUNCTION:
                source = operator.attrgetter(f'{FUNCTION}.parameters.{p.name}')
            else:
                source = p

            # use Shared/FunctionParameter value as fallback
            try:
                value = owner.initial_shared_parameters[corresponding_parameter_component_name][p.name]
            except (KeyError, TypeError):
                value = None

            # if parameter value on actual Parameter was specified or there is
            # no Shared/FunctionParameter value, use the actual Parameter default
            if p._user_specified or value is None:
                if p.spec is not None:
                    value = p.spec
                else:
                    value = p.default_value

            if add_suffix:
                explicit_name = ParameterPortList._get_explicit_name(
                    p.name,
                    corresponding_parameter_component_name
                )
            else:
                explicit_name = p.name

            _instantiate_parameter_port(
                owner,
                p.name,
                value,
                context=context,
                function=corresponding_parameter_component,
                source=source,
                explicit_name=explicit_name
            )

    for p in owner.parameters:
        if (
            not skip_parameter_port(p)
            and p.name not in owner.parameter_ports.names
        ):
            if p.spec is not None:
                value = p.spec
            else:
                value = p.default_value

            _instantiate_parameter_port(
                owner,
                p.name,
                value,
                context=context,
                function=function,
                source=p
            )

    owner.parameter_ports.sort(key=lambda port: port.name)

def _instantiate_parameter_port(
    owner,
    param_name,
    param_value,
    context,
    function=None,
    source=None,
    explicit_name=None
):
    """Call _instantiate_port for allowable params, to instantiate a ParameterPort for it

    Include ones in function.parameters
    Exclude if it is a:
        ParameterPort that already exists
        non-numeric value (including NotImplemented, False or True)
            unless it is:
                a tuple (could be one specifying Modulatory Component)
                a dict with the name FUNCTION_PARAMS (otherwise exclude)
        function or method
            IMPLEMENTATION NOTE: FUNCTION_RUNTIME_PARAM_NOT_SUPPORTED
            (this is because self.defaults.function could be a class rather than an bound method;
            i.e., not yet instantiated;  could be rectified by assignment in _instantiate_function)
    # FIX: UPDATE WITH MODULATION_MODS
    # FIX:    CHANGE TO IntegratorFunction FUnction ONCE LearningProjection MODULATES ParameterPort Function:
    If param_name is FUNCTION_PARAMS and param is a matrix (presumably for a MappingProjection)
        modify ParameterPort's function to be LinearCombination (rather Linear which is the default)
    """
    from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import _is_modulatory_spec
    from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base

    def _get_tuple_for_single_item_modulatory_spec(obj, name, value):
        """Return (<default param value>, <modulatory spec>) for modulatory spec
        """
        try:
            param_default_value = getattr(obj.defaults, name)
            # Only assign default value if it is not None
            if param_default_value is not None:
                return (param_default_value, value)
            else:
                return value
        except AttributeError:
            raise ParameterPortError("Unrecognized specification for {} paramater of {} ({})".
                                      format(param_name, owner.name, param_value))

    if explicit_name is None:
        explicit_name = param_name

    # EXCLUSIONS:

    # # Skip if ParameterPort already exists
    # if param_name in owner.ParameterPorts:
    #     return

    if param_value is NotImplemented:
        return
    # Allow numerics but omit booleans (which are treated by is_numeric as numerical)
    if is_numeric(param_value) and not isinstance(param_value, bool):
        pass
    # Only allow a FUNCTION_PARAMS dict
    elif isinstance(param_value, (ReadOnlyOrderedDict, dict)) and param_name == FUNCTION_PARAMS:
        pass
    # Allow ModulatoryProjection
    elif isinstance(param_value, Projection):
        if isinstance(param_value, ModulatoryProjection_Base):
            pass
        else:
            return
    # Allow Projection class
    elif inspect.isclass(param_value) and issubclass(param_value, Projection):
        if issubclass(param_value, (ModulatoryProjection_Base)):
            pass
        else:
            return

    elif _is_modulatory_spec(param_value, include_matrix_spec=False) and not isinstance(param_value, tuple):
        # If parameter is a single Modulatory specification (e.g., ControlSignal, or CONTROL, etc.)
         #  try to place it in a tuple (for interpretation by _parse_port_spec) using default value as 1st item
        #   (note: exclude matrix since it is allowed as a value specification but not a projection reference)
        try:
            param_value = _get_tuple_for_single_item_modulatory_spec(function, param_name, param_value)
        except ParameterPortError:
            param_value = _get_tuple_for_single_item_modulatory_spec(owner, param_name, param_value)

    # Allow tuples (could be spec that includes a Projection or Modulation)
    elif isinstance(param_value, tuple):
        # # FIX: EXTRACT VALUE HERE (AS IN Component.__init__?? [4/18/17]
        # param_value = owner._get_param_value_from_tuple(param_value)
        pass
    # Allow if it is a keyword for a parameter
    elif isinstance(param_value, str) and param_value in parameter_keywords:
        pass
    # Exclude function (see docstring above)
    elif param_name == FUNCTION:
        return
    # (7/19/17 CW) added this if statement below while adding `hetero` and `auto` and AutoAssociativeProjections: this
    # allows `hetero` to be specified as a matrix, while still generating a ParameterPort
    elif isinstance(param_value, np.ndarray) or isinstance(param_value, np.matrix):
        pass
    # allow function parameters
    elif param_name in function.parameters.names():
        pass
    # Exclude all others
    else:
        return

    # Assign ParameterPorts to Component for parameters of its function (function_params), except for ones that are:
    #    - another component
    #    - a function or method
    #    - have a value of None (see IMPLEMENTATION_NOTE below)
    #    - they have the same name as another parameter of the component (raise exception for this)

    # IMPLEMENTATION NOTE:
    # The following is necessary since, if ANY parameters of a function are specified, entries are made
    #    in the FUNCTION_PARAMS dict of its owner for ALL of the function's params;  however, their values
    #    will be set to None (and there may not be a way to determine a
    #    default; e.g., the length of the array for the weights or exponents params for LinearCombination).
    #    Therefore, None will be passed as the reference_value, which will cause validation of the
    #    ParameterPort's function (in _instantiate_function()) to fail.
    #  Current solution is to simply not instantiate a ParameterPort for any function_param that has
    #    not been explicitly specified
    if param_value is None:
        return

    if not _is_legal_param_value(owner, param_value):
        return

    elif (_is_modulatory_spec(param_value, include_matrix_spec=False)
            and not isinstance(param_value, tuple)):
        # If parameter is a single Modulatory specification (e.g., ControlSignal, or CONTROL, etc.)
        # try to place it in a tuple (for interpretation by _parse_port_spec) using default value as 1st item
        #   (note: exclude matrix since it is allowed as a value specification vs. a projection reference)
        try:
            param_value = _get_tuple_for_single_item_modulatory_spec(
                function,
                param_name,
                param_value
            )
        except ParameterPortError:
            param_value = _get_tuple_for_single_item_modulatory_spec(
                owner,
                param_name,
                param_value
            )

    # # FIX: 10/3/17 - ??MOVE THIS TO _parse_port_specific_specs ----------------
    # # Use param_value as constraint
    # # IMPLEMENTATION NOTE:  need to copy, since _instantiate_port() calls _parse_port_value()
    # #                       for constraints before port_spec, which moves items to subdictionaries,
    # #                       which would make them inaccessible to the subsequent parse of port_spec
    from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import ModulatorySignal
    from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
    if (
        is_iterable(param_value)
        and any(isinstance(item, (ModulatorySignal, ModulatoryProjection_Base, ModulatoryMechanism_Base)) for item in param_value)
    ):
        reference_value = param_value
    else:
        reference_value = deepcopy(param_value)

    # Assign parameterPort for function_param to the component
    port = _instantiate_port(
        owner=owner,
        port_type=ParameterPort,
        name=explicit_name,
        port_spec=param_value,
        reference_value=reference_value,
        reference_value_name=param_name,
        params=None,
        context=context
    )
    if port:
        owner._parameter_ports[explicit_name] = port
        # will be parsed on assignment of function
        # FIX: if the function is manually changed after assignment,
        # FIX: the source will remain pointing to the original Function
        port.source = source
        # if the source parameter is not added here, we can't reference
        # a ParameterPort by Parameter
        owner.parameter_ports.parameter_mapping[source] = port

    return port


def _is_legal_param_value(owner, value):

    from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import _is_control_spec
    from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import _is_gating_spec

    # LEGAL PARAMETER VALUES:

    # # lists, arrays or numeric values
    if is_value_spec(value):
        return True

    # tuple, first item of which is a legal parameter value
    #     note: this excludes (param_name, Mechanism) tuples used to specify a ParameterPort
    #           (e.g., if specified for the control_signals param of ControlMechanism)
    if isinstance(value, tuple):
        if _is_legal_param_value(owner, value[0]):
            return True

    if isinstance(value, dict) and VALUE in value:
        return True

    if _is_control_spec(value) or _is_gating_spec(value):
        return True

    # keyword that resolves to one of the above
    if get_param_value_for_keyword(owner, value) is not None:
        return True

    # Assignment of ParameterPort for Component objects, function or method are not currently supported
    if isinstance(value, (types.FunctionType, types.MethodType, Component)):
        return False


def _get_parameter_port(sender_owner, sender_type, param_name, component):
    """Return ParameterPort for named parameter of a Mechanism requested by owner
    """

    # Validate that component is a Mechanism or Projection
    if not isinstance(component, (Mechanism, Projection)):
        raise ParameterPortError("Request for {} of a component ({}) that is not a {} or {}".
                                  format(PARAMETER_PORT, component, MECHANISM, PROJECTION))

    try:
        return component._parameter_ports[param_name]
    except KeyError:
        # Check that param (named by str) is an attribute of the Mechanism
        if not (hasattr(component, param_name) or hasattr(component.function, param_name)):
            raise ParameterPortError("{} (in specification of {}  {}) is not an attribute "
                                        "of {} or its function"
                                        .format(param_name, sender_type, sender_owner.name, component))
        # Check that the Mechanism has a ParameterPort for the param
        if param_name not in component._parameter_ports.names:
            raise ParameterPortError("There is no ParameterPort for the parameter ({}) of {} "
                                        "specified in {} for {}".
                                        format(param_name, component.name, sender_type, sender_owner.name))
