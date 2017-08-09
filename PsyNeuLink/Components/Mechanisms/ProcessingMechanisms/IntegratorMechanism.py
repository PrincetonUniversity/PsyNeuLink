# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  IntegratorMechanism *************************************************

"""
Overview
--------

An IntegratorMechanism integrates its input, possibly based on its prior values.  The input can be a single
scalar value or an array of scalars (list or 1d np.array).  If it is a list or array, then each value is
independently integrated.  The default function (`Integrator`) can be parametrized to implement either a simple
increment rate, additive accumulator, or an (exponentially weighted) time-averaging of its input.  It can also be
assigned a custom function.

.. _IntegratorMechanism_Creation:

Creating an IntegratorMechanism
-------------------------------

An IntegratorMechanism can be created directly by calling its constructor, or using the
`mechanism() <Mechanism.mechanism>` function and specifying *INTEGRATOR_MECHANISM* as its **mech_spec**
argument.  Its function is specified in the **function** argument, which can be parametrized by calling its
constructor with parameter values::

    my_time_averaging_mechanism = IntegratorMechanism(function=AdaptiveIntegrator(rate=0.5))

The **default_variable** argument specifies the format of its input (i.e., whether it is a single scalar or an
array), as well as the value to use if none is provided when mechanism is executed.  Alternatively, the **size**
argument can be used to specify the length of the array, in which case it will be initialized with all zeros.

.. _IntegratorMechanism_Structure

Structure
---------

An IntegratorMechanism has a single `InputState`, the `value <InputState.InputState.value>` of which is
used as the  `variable <IntegratorMechanism.variable>` for its `function <IntegratorMechanism.function>`.
The default for `function <IntegratorMechanism.function>` is `AdaptiveIntegrator(rate=0.5)`. However,
a custom function can also be specified,  so long as it takes a numeric value, or a list or np.ndarray of numeric
values as its input, and returns a value of the same type and format.  The mechanism has a single `OutputState`,
the `value <OutputState.OutputState.value>` of which is assigned the result of  the call to the Mechanism's
`function  <IntegratorMechanism.function>`.

.. _IntegratorMechanism_Execution

Execution
---------

When an IntegratorMechanism is executed, it carries out the specified integration, and assigns the
result to the `value <IntegratorMechanism.value>` of its `primary OutputState <OutputState_Primary>`.  For the default
function (`Integrator`), if the value specified for **default_variable** is a list or array, or **size** is greater
than 1, each element of the array is independently integrated.  If its `rate <Integrator.rate>` parameter is a
single value,  that rate will be used for integrating each element.  If the `rate <Integrator.rate>` parameter is a
list or array, then each element will be used as the rate for the corresponding element of the input (in this case,
`rate <Integrator.rate>` must be the same length as the value specified for **default_variable** or **size**).


.. _IntegratorMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc

from PsyNeuLink.Components import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Globals.Keywords import INTEGRATOR_MECHANISM, OUTPUT_STATES, PREDICTION_MECHANISM_OUTPUT, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Scheduling.TimeScale import TimeScale

# IntegratorMechanism parameter keywords:
DEFAULT_RATE = 0.5

class IntegratorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class IntegratorMechanism(ProcessingMechanism_Base):
    """
    IntegratorMechanism(                            \
    default_variable=None,                               \
    size=None,                                              \
    function=AdaptiveIntegrator(rate=0.5), \
    time_scale=TimeScale.TRIAL,                             \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Subclass of `ProcessingMechanism` that integrates its input.

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
        Class methods:
            None

        MechanismRegistry:
           All instances of SigmoidLayer are registered in MechanismRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray
        the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <IntegratorMechanism.variable>` for
        `function <IntegratorMechanism.function>`, and the `primary outputState <OutputState_Primary>` of the
        mechanism.

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.

    function : IntegratorFunction : default Integrator
        specifies the function used to integrate the input.  Must take a single numeric value, or a list or np.array
        of values, and return one of the same form.

    time_scale :  TimeScale : TimeScale.TRIAL
        specifies whether the mechanism is executed on the TIME_STEP or TRIAL time scale.
        This must be set to `TimeScale.TIME_STEP` for the :keyword:`rate` parameter to have an effect.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the mechanism, parameters for its `function <IntegratorMechanism.function>`, and/or a custom function and its
        parameters.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    name : str : default IntegratorMechanism-<index>
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

    name : str : default IntegratorMechanism-<index>
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

    componentType = INTEGRATOR_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'IntegratorMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    # Sets template for variable (input)
    variableClassDefault = [[0]]

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        # TIME_SCALE: TimeScale.TRIAL,
        OUTPUT_STATES:[PREDICTION_MECHANISM_OUTPUT]

    })

    from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=AdaptiveIntegrator(rate=0.5),
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        if default_variable is None and size is None:
            default_variable = self.variableClassDefault

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        # self.variableClassDefault = default_variable or [[0]]
        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        # if default_variable is NotImplemented:
        #     default_variable = SigmoidLayer_DEFAULT_NET_INPUT

        # self.size = size

        super(IntegratorMechanism, self).__init__(variable=default_variable,
                                                  size=size,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED

    # MODIFIED 6/2/17 NEW:
    @property
    def previous_value(self):
        return self.function_object.previous_value
    # MODIFIED 6/2/17 END


