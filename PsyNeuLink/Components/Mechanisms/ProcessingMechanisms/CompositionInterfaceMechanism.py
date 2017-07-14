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
Composition's origin mechanism via a projection.

.. _CompositionInterfaceMechanism_Creation:

Creating an CompositionInterfaceMechanism
-------------------------------

A CompositionInterfaceMechanism is created automatically when an Origin mechanism is identified in a Composition.

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

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.Functions.Function import Linear

class CompositionInterfaceMechanism(ProcessingMechanism_Base):
    """
    CompositionInterfaceMechanism(                            \
    default_input_value=None,                               \
    size=None,                                              \
    function=Linear(slope = 1.0, intercept = 0.0), \
    time_scale=TimeScale.TRIAL,                             \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Implements the CompositionInterfaceMechanism subclass of Mechanism.

    COMMENT:
        Description:
            - DOCUMENT:

        Class attributes:
            + componentType (str): SigmoidLayer
            + classPreference (PreferenceSet): SigmoidLayer_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + variableClassDefault (value):  SigmoidLayer_DEFAULT_BIAS
            + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL,
                                          FUNCTION_PARAMS:{kwSigmoidLayer_Unitst: kwSigmoidLayer_NetInput
                                                                     kwSigmoidLayer_Gain: SigmoidLayer_DEFAULT_GAIN
                                                                     kwSigmoidLayer_Bias: SigmoidLayer_DEFAULT_BIAS}}
            + paramNames (dict): names as above

        Class methods:
            None

        MechanismRegistry:
           All instances of SigmoidLayer are registered in MechanismRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT

    Arguments
    ---------

    default_input_value : number, list or np.ndarray
        the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism.Mechanism_Base.execute>` or `run <Mechanism.Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <CompositionInterfaceMechanism.variable>` for
        `function <CompositionInterfaceMechanism.function>`, and the `primary outputState <OutputState_Primary>` of the
        mechanism.

    size : int, list or np.ndarray of ints
        specifies default_input_value as array(s) of zeros if **default_input_value** is not passed as an argument;
        if **default_input_value** is specified, it takes precedence over the specification of **size**.

    function : IntegratorFunction : default Integrator
        specifies the function used to integrate the input.  Must take a single numeric value, or a list or np.array
        of values, and return one of the same form.

    time_scale :  TimeScale : TimeScale.TRIAL
        specifies whether the mechanism is executed on the TIME_STEP or TRIAL time scale.
        This must be set to `TimeScale.TIME_STEP` for the :keyword:`rate` parameter to have an effect.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the mechanism, parameters for its `function <CompositionInterfaceMechanism.function>`, and/or a custom function and its
        parameters.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    name : str : default CompositionInterfaceMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------
    variable : value: default
        the input to Mechanism's ``function``.

    time_scale :  TimeScale : defaultTimeScale.TRIAL
        specifies whether the Mechanism is executed on the TIME_STEP or TRIAL time scale.

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

    # Sets template for variable (input)
    variableClassDefault = [[0]]

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        # TIME_SCALE: TimeScale.TRIAL,
        OUTPUT_STATES:[PREDICTION_MECHANISM_OUTPUT]

    })

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
            default_input_value = self.variableClassDefault

        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        super(CompositionInterfaceMechanism, self).__init__(variable=default_input_value,
                                                  size=size,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)



