# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  CompositionInterfaceMechanism *************************************************

"""
Overview
--------

A CompositionInterfaceMechanism stores an input from the outside world so that the input can be delivered to a
Composition's `ORIGIN` Mechanism via a Projection.

.. _CompositionInterfaceMechanism_Creation:

Creating an CompositionInterfaceMechanism
-------------------------------

A CompositionInterfaceMechanism is created automatically when an `ORIGIN` Mechanism is identified in a Composition. When
created, the CompositionInterfaceMechanism's OutputState is set directly by the Composition. This Mechanism should never
be executed, and should never be created by a user.

.. _CompositionInterfaceMechanism_Structure

Structure
---------

---
.. _CompositionInterfaceMechanism_Execution

Execution
---------

---

.. _CompositionInterfaceMechanism_Class_Reference:

Class Reference
---------------

"""

import typecheck as tc

from PsyNeuLink.Components.Functions.Function import Linear
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Globals.Keywords import COMPOSITION_INTERFACE_MECHANISM, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Scheduling.TimeScale import TimeScale


class CompositionInterfaceMechanism(ProcessingMechanism_Base):
    """
    CompositionInterfaceMechanism(                            \
    default_input_value=None,                               \
    size=None,                                              \
    function=Linear(slope = 1.0, intercept = 0.0), \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Implements the CompositionInterfaceMechanism subclass of Mechanism.

    Arguments
    ---------

    default_input_value : number, list or np.ndarray
        the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <CompositionInterfaceMechanism.variable>` for
        `function <CompositionInterfaceMechanism.function>`, and the `primary outputState <OutputState_Primary>` of the
        Mechanism.

    size : int, list or np.ndarray of ints
        specifies default_input_value as array(s) of zeros if **default_input_value** is not passed as an argument;
        if **default_input_value** is specified, it takes precedence over the specification of **size**.

    function : IntegratorFunction : default Integrator
        specifies the function used to integrate the input.  Must take a single numeric value, or a list or np.array
        of values, and return one of the same form.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the `Mechanism <Mechanism>`, parameters for its `function <CompositionInterfaceMechanism.function>`, and/or a
        custom function and its parameters.  Values specified for parameters in the dictionary override any assigned
        to those parameters in arguments of the constructor.

    name : str : default CompositionInterfaceMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------
    variable : value: default
        the input to Mechanism's ``function``.

    name : str : default CompositionInterfaceMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Mechanism;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentType = COMPOSITION_INTERFACE_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'CompositionInterfaceMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    class ClassDefaults(ProcessingMechanism_Base.ClassDefaults):
        # Sets template for variable (input)
        variable = [[0]]

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({})
    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 size=None,
                 function = Linear(slope = 1, intercept=0.0),
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        if default_input_value is None and size is None:
            default_input_value = self.ClassDefaults.variable

        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        super(CompositionInterfaceMechanism, self).__init__(variable=default_input_value,
                                                  size=size,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)



