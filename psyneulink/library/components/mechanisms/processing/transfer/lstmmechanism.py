# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ****************************************  LSTMMechanism *************************************************

import numpy as np
import typecheck as tc

from collections.abc import Iterable

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.transferfunctions import LSTM, Logistic, Tanh
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.globals.keywords import IDENTITY_MATRIX, LSTM_MECHANISM, OWNER_VALUE, RESULT
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.components.ports.outputport import OutputPort

__all__ = [
    'LSTMMechanism'
]

RECURRENT_INPUT_PORT_NAME = "RECURRENT_INPUT"
CELL_STATE_INPUT_PORT_NAME = 'CELL_STATE_INPUT'
CELL_STATE_OUTPUT_PORT_NAME = 'CELL_STATE_OUTPUT'
EXTERNAL = 'EXTERNAL'

class LSTMMechanism(ProcessingMechanism):
    componentType = LSTM_MECHANISM

    class Parameters(ProcessingMechanism.Parameters):
        i_input_matrix = Parameter(modulable=True, function_parameter=True)
        i_hidden_matrix = Parameter(modulable=True, function_parameter=True)
        i_gate_func = Parameter(default_value=Logistic(), function_parameter=True)

        f_input_matrix = Parameter(modulable=True, function_parameter=True)
        f_hidden_matrix = Parameter(modulable=True, function_parameter=True)
        f_gate_func = Parameter(default_value=Logistic(), function_parameter=True)

        g_input_matrix = Parameter(modulable=True, function_parameter=True)
        g_hidden_matrix = Parameter(modulable=True, function_parameter=True)
        g_gate_func = Parameter(default_value=Tanh(), function_parameter=True)

        o_input_matrix = Parameter(modulable=True, function_parameter=True)
        o_hidden_matrix = Parameter(modulable=True, function_parameter=True)
        o_gate_func = Parameter(default_value=Logistic(), function_parameter=True)

        h_gate_func = Parameter(default_value=Tanh(), function_parameter=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports: tc.optional(tc.any(list, dict)) = None,
                 initial_value=None,
                 output_ports: tc.optional(tc.any(str, Iterable)) = None,
                 params=None,
                 name=None,
                 **kwargs):

        if default_variable is not None:
            size = default_variable.shape
            assert len(default_variable) == 3, "Must specify default variables for input, hidden state, and cell state!"
        else:
            assert size is not None and len(size) == 2, "Must only specify input and hidden sizes!"
            size = (*size, size[1])
            default_variable = np.array([np.zeros(size[0]), np.zeros(size[1]), np.zeros(size[1])])

        # FIXME: Need to store these values, since self.size doesn't seem to be getting set as intended to a tuple (seems like a bug)
        self.input_size = size[0]
        self.hidden_size = size[1]

        recurrent_input_port = InputPort(reference_value=np.zeros(size[1:2]), variable=np.zeros(size[1:2]), name=RECURRENT_INPUT_PORT_NAME)
        cell_state_input_port = InputPort(reference_value=np.zeros(size[1:2]), variable=np.zeros(size[1:2]), name=CELL_STATE_INPUT_PORT_NAME)
        cell_state_output_port = OutputPort(reference_value=np.zeros(size[1:2]), variable=(OWNER_VALUE, 1), name=CELL_STATE_OUTPUT_PORT_NAME)

        if input_ports is None or input_ports == EXTERNAL:
            input_ports = [EXTERNAL, recurrent_input_port, cell_state_input_port]

        if output_ports is None or output_ports == RESULT:
            output_ports = [RESULT, cell_state_output_port]

        function = LSTM(default_variable=default_variable, owner=self, params=kwargs)

        super().__init__(
            default_variable=default_variable,
            size=size,
            input_ports=input_ports,
            initial_value=initial_value,
            output_ports=output_ports,
            params=params,
            name=name,
            function=function,
            **kwargs
        )

        # Set the recurrent input port as internal only, since it shouldn't connect to the composition's input_cim
        self.input_ports[RECURRENT_INPUT_PORT_NAME].internal_only = True
        self.input_ports[CELL_STATE_INPUT_PORT_NAME].internal_only = True

        hidden_recurrent_proj = MappingProjection(sender=self.output_port,
                                                  receiver=recurrent_input_port,
                                                  matrix=IDENTITY_MATRIX,
                                                  name=self.name + ' hidden recurrent projection',
                                                  learnable=False)

        cell_recurrent_proj = MappingProjection(sender=cell_state_output_port,
                                                receiver=cell_state_input_port,
                                                matrix=IDENTITY_MATRIX,
                                                name=self.name + ' cell recurrent projection',
                                                learnable=False)

        self.aux_components.append(hidden_recurrent_proj)
        self.aux_components.append(cell_recurrent_proj)

        # HACK: Manually reset default variable for output port (is set in initialization, but this means initial state gets overwritten)
        self.output_ports.value = default_variable[1:]

    @property
    def initial_hidden_state(self):
        return self.defaults.value[0]

    @property
    def initial_cell_state(self):
        return self.defaults.value[1]
