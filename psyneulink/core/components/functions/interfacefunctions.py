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

* InterfacePortMap

"""

import numpy as np
import typecheck as tc
import warnings

from psyneulink.core.components.functions.function import Function_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    FUNCTION_OUTPUT_TYPE_CONVERSION, PARAMETER_PORT_PARAMS, PORT_MAP_FUNCTION, TRANSFER_FUNCTION_TYPE, \
    PREFERENCE_SET_NAME
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import \
    PreferenceEntry, PreferenceLevel, is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.utilities import convert_to_np_array


__all__ = ['InterfaceFunction', 'InterfacePortMap']

class InterfaceFunction(Function_Base):
    """Simple functions for CompositionInterfaceMechanisms
    """
    componentType = TRANSFER_FUNCTION_TYPE


class InterfacePortMap(InterfaceFunction):
    """
    Identity(                \
             default_variable, \
             params=None,      \
             owner=None,       \
             name=None,        \
             prefs=None        \
            )

    .. _Identity:

    Returns `variable <InterfacePortMap.variable>`.

    Arguments
    ---------

    default_variable : number or np.array : default class_defaults.variable
        specifies a template for the value to be transformed.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).
    """

    componentName = PORT_MAP_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'LinearClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(InterfaceFunction.Parameters):
        """
            Attributes
            ----------

                corresponding_input_port
                    see `corresponding_input_port <InterfacePortMap.corresponding_input_port>`

                    :default value: None
                    :type:
        """
        corresponding_input_port = Parameter(
            None,
            structural=True,
            stateful=False,
            loggable=False
        )

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 corresponding_input_port=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            corresponding_input_port=corresponding_input_port,
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
        Return: The item of `value <InterfacePortMap.value>` whose index corresponds to the index of
        `corresponding_input_port <InterfacePortMap.corresponding_input_port>` in `input_ports
        <InterfacePortMap.input_ports>`

        Arguments
        ---------

        variable : number or np.array : default class_defaults.variable
           a single value or array to be transformed.

        corresponding_input_port : InputPort : default None
            the InputPort on the owner CompositionInterfaceMechanism to which this OutputPort corresponds

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        The item of `value <InterfacePortMap.value>` whose index corresponds to the index of
        `corresponding_input_port <InterfacePortMap.corresponding_input_port>` in `input_ports
        <InterfacePortMap.input_ports>`

        """
        index = self.corresponding_input_port.position_in_mechanism

        if self.corresponding_input_port.owner.parameters.value._get(context) is not None:

            # If CIM's variable does not match its value, then a new pair of ports was added since the last execution
            input_values = convert_to_np_array(
                self.corresponding_input_port.owner.get_input_values(context)
            )
            if not np.shape(input_values) == np.shape(self.corresponding_input_port.owner.parameters.value._get(context)):
                return self.corresponding_input_port.owner.defaults.variable[index]

            # If the variable is 1D (e.g. [0. , 0.], NOT [[0. , 0.]]), and the index is 0, then return whole variable
            # np.atleast_2d fails in cases like var = [[0., 0.], [0.]] (transforms it to [[[0., 0.], [0.]]])
            if index == 0:
                if not isinstance(variable[0], (list, np.ndarray)):
                    return variable
            return variable[index]
        # CIM value = None, use CIM's default variable instead
        return self.corresponding_input_port.owner.defaults.variable[index]

    def _get_input_struct_type(self, ctx):
        #FIXME: Workaround for CompositionInterfaceMechanism that
        #       does not update its default shape
        from psyneulink import CompositionInterfaceMechanism
        if hasattr(self.owner, 'owner') and isinstance(self.owner.owner, CompositionInterfaceMechanism):
            warnings.warn("Shape mismatch: {} input: {} vs. ({}, MV:{})".format(
                self, self.defaults.variable, self.owner._variable_spec,
                self.owner.owner.function.defaults.value))
            return ctx.get_output_struct_type(self.owner.owner.function)
        return ctx.get_input_struct_type(super())

    def _get_output_struct_type(self, ctx):
        index = self.corresponding_input_port.position_in_mechanism
        input_type = ctx.get_input_struct_type(self)
        return input_type.elements[index]

    def _gen_llvm_function_body(self, ctx, builder, _1, _2, arg_in, arg_out, *, tags:frozenset):
        index = self.corresponding_input_port.position_in_mechanism
        val = builder.load(builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(index)]))
        builder.store(val, arg_out)
        return builder
