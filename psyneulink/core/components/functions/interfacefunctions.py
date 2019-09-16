#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *********************************************  INTERFACE FUNCTIONS ***************************************************

"""

* InterfaceStateMap

"""

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.function import Function_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    FUNCTION_OUTPUT_TYPE_CONVERSION, PARAMETER_STATE_PARAMS, STATE_MAP_FUNCTION, TRANSFER_FUNCTION_TYPE, \
    kwPreferenceSetName
from psyneulink.core.globals.preferences.componentpreferenceset import \
    PreferenceEntry, PreferenceLevel, is_pref_set, kpReportOutputPref


__all__ = ['InterfaceFunction', 'InterfaceStateMap']

class InterfaceFunction(Function_Base):
    """Simple functions for CompositionInterfaceMechanisms
    """
    componentType = TRANSFER_FUNCTION_TYPE


class InterfaceStateMap(InterfaceFunction):
    """
    Identity(                \
             default_variable, \
             params=None,      \
             owner=None,       \
             name=None,        \
             prefs=None        \
            )

    .. _Identity:

    Returns `variable <InterfaceStateMap.variable>`.

    Arguments
    ---------

    default_variable : number or np.array : default class_defaults.variable
        specifies a template for the value to be transformed.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        contains value to be transformed.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = STATE_MAP_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'LinearClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: True,
        PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 corresponding_input_state=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(corresponding_input_state=corresponding_input_state,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         )

        # self.functionOutputType = None

    def _function(
        self,
        variable=None,
        context=None,
        params=None,

    ):
        """
        Return: The item of `value <InterfaceStateMap.value>` whose index corresponds to the index of
        `corresponding_input_state <InterfaceStateMap.corresponding_input_state>` in `input_states
        <InterfaceStateMap.input_states>`

        Arguments
        ---------

        variable : number or np.array : default class_defaults.variable
           a single value or array to be transformed.

        corresponding_input_state : InputState : default None
            the InputState on the owner CompositionInterfaceMechanism to which this OutputState corresponds

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        The item of `value <InterfaceStateMap.value>` whose index corresponds to the index of
        `corresponding_input_state <InterfaceStateMap.corresponding_input_state>` in `input_states
        <InterfaceStateMap.input_states>`

        """
        index = self.corresponding_input_state.position_in_mechanism

        if self.corresponding_input_state.owner.parameters.value._get(context) is not None:

            # If CIM's variable does not match its value, then a new pair of states was added since the last execution
            if not np.shape(self.corresponding_input_state.owner.get_input_values(context)) == np.shape(self.corresponding_input_state.owner.parameters.value._get(context)):
                return self.corresponding_input_state.owner.defaults.variable[index]

            # If the variable is 1D (e.g. [0. , 0.], NOT [[0. , 0.]]), and the index is 0, then return whole variable
            # np.atleast_2d fails in cases like var = [[0., 0.], [0.]] (transforms it to [[[0., 0.], [0.]]])
            if index == 0:
                if not isinstance(variable[0], (list, np.ndarray)):
                    return variable
            return variable[index]
        # CIM value = None, use CIM's default variable instead
        return self.corresponding_input_state.owner.defaults.variable[index]

    def _get_input_struct_type(self, ctx):
        #FIXME: Workaround for CompositionInterfaceMechanism that
        #       does not update its defaults shape
        from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
        if hasattr(self.owner, 'owner') and isinstance(self.owner.owner, CompositionInterfaceMechanism):
            return ctx.get_output_struct_type(self.owner.owner.function)
        return ctx.get_input_struct_type(super())

    def _gen_llvm_function_body(self, ctx, builder, _1, _2, arg_in, arg_out):
        index = self.corresponding_input_state.position_in_mechanism
        val = builder.load(builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(index)]))
        builder.store(val, arg_out)
        return builder
