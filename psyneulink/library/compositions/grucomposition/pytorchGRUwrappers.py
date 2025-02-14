# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for GRUComposition"""

import torch
import numpy as np
from typing import Union, Optional, Literal

from psyneulink.core.components.functions.stateful.statefulfunction import StatefulFunction
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper, \
    PytorchProjectionWrapper, PytorchFunctionWrapper
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import ALL, INPUTS, OUTPUTS

__all__ = ['PytorchGRUCompositionWrapper', 'GRU_NODE_NAME', 'TARGET_NODE_NAME']

GRU_NODE_NAME = 'PYTORCH GRU NODE'
TARGET_NODE_NAME = 'GRU TARGET NODE'

class PytorchGRUCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for GRUComposition as a Pytorch Module
    Manage the exchange of the Composition's Projection `Matrices <MappingProjection_Matrix>`
    and the Pytorch GRU Module's parameters, and return its output value.
    """
    def _instantiate_pytorch_mechanism_wrappers(self, composition, device, context):
        """Instantiate PytorchMechanismWrapper for GRU Node"""
        node = composition.gru_mech
        pytorch_node = PytorchGRUMechanismWrapper(node, self, 0, device, context)
        self.nodes_map[node] = pytorch_node
        self.wrapped_nodes.append(pytorch_node)
        if not composition.is_nested:
            node._is_input = True

    def _instantiate_pytorch_projection_wrappers(self, composition, device, context):
        """Assign PytorchProjectionWrapper's parameters to those of GRU Node"""
        if len(self.wrapped_nodes) == 1:
            self.parameters = self.wrapped_nodes[0].function.function.parameters
        else:
            if not len(self.wrapped_nodes):
                assert False, \
                    (f"PROGRAM ERROR: PytorchGRUCompositionWrapper has no wrapped nodes; should have one for "
                     f"'PYTORCH GRU NODE'.")
            else:
                extra_nodes = [node for node in self.wrapped_nodes
                               if node.name != 'PytorchMechanismWrapper[PYTORCH GRU NODE]']
                assert False, \
                    (f"PROGRAM ERROR: Somehow an extra node or more snuck into PytorchGRUCompositionWrapper; "
                     f"should only have one for 'PYTORCH GRU NODE', but also has: {extra_nodes}.")

    @handle_external_context()
    def forward(self, inputs, optimization_num, context=None)->dict:
        """Forward method of the model for PyTorch modes
        Returns a dictionary {output_node:value} with the output value for the module in case it is run as a
        standalone Composition; otherwise, those will be ignored and the outputs will be used by the aggregate_afferents
        method(s) of the other node(s) that receive Projections from the GRUComposition.
        """
        # Reshape iput for GRU module (from float64 to float32
        inputs = torch.tensor(np.array(inputs[self._composition.input_node]).astype(np.float32))
        hidden_state = self._composition.hidden_state
        gru_node = self._composition.gru_mech
        output, self.hidden_state = gru_node.function(inputs, hidden_state)
        # Assign output to the OUTPUT Node of the GRUComposition
        self._composition.output_node.parameters.value._set(output.detach().cpu().numpy(), context)
        return {self._composition.output_node: output}

    def copy_weights_to_psyneulink(self, context=None):
        for projection, pytorch_rep in self.projections_map.items():
            matrix = pytorch_rep.matrix.detach().cpu().numpy()
            projection.parameters.matrix._set(matrix, context)
            projection.parameters.matrix._set(matrix, context)
            projection.parameter_ports['matrix'].parameters.value._set(matrix, context)

    def log_weights(self):
        for proj_wrapper in self.projection_wrappers:
            proj_wrapper.log_matrix()

    def copy_node_variables_to_psyneulink(self, nodes:Optional[Union[list,Literal[ALL, INPUTS]]]=ALL, context=None):
        """Copy input to Pytorch nodes to variable of AutodiffComposition nodes.
        IMPLEMENTATION NOTE:  list included in nodes arg to allow for future specification of specific nodes to copy
        """
        if nodes == ALL:
            nodes = self.nodes_map.items()
        for pnl_node, pytorch_node in nodes:
            # First get variable in numpy format
            if isinstance(pytorch_node.input, list):
                variable = np.array([val.detach().cpu().numpy() for val in pytorch_node.input], dtype=object)
            else:
                variable = pytorch_node.input.detach().cpu().numpy()
            # Set pnl_node's value to value
            pnl_node.parameters.variable._set(variable, context)

    def copy_node_values_to_psyneulink(self, nodes:Optional[Union[list,Literal[ALL, OUTPUTS]]]=ALL, context=None):
        """Copy output of Pytorch nodes to value of AutodiffComposition nodes.
        IMPLEMENTATION NOTE:  list included in nodes arg to allow for future specification of specific nodes to copy
        """
        if nodes == ALL:
            nodes = self.nodes_map.items()
        # elif nodes == OUTPUTS:
        #     nodes = [(node, self.nodes_map[node]) for node in self._composition.get_output_nodes()]

        def update_autodiff_all_output_values():
            """Update autodiff's output_values by executing its output_CIM's with pytorch_rep all_output_values"""
            if self.all_output_values:
                self._composition.output_CIM.execute(self.all_output_values, context=context)

        # Allow selective updating of just autodiff.output_values if specified
        if nodes == OUTPUTS:
            update_autodiff_all_output_values()
            return

        for pnl_node, pytorch_node in nodes:
            # Update each node's value with the output of the corresponding wrappter in the PyTorch representation
            if pytorch_node.output is None:
                assert pytorch_node.exclude_from_gradient_calc, \
                    (f"PROGRAM ERROR: Value of PyTorch wrapper for {pnl_node.name} is None during forward pass, "
                     f"but it is not excluded from gradient calculation.")
                continue
            # First get value in numpy format
            if isinstance(pytorch_node.output, list):
                value = np.array([val.detach().cpu().numpy() for val in pytorch_node.output], dtype=object)
            else:
                value = pytorch_node.output.detach().cpu().numpy()

            # Set pnl_node's value to value
            pnl_node.parameters.value._set(value, context)

            # If pnl_node's function is Stateful, assign value to its previous_value parameter
            #   so that if Python implementation is run it picks up where PyTorch execution left off
            if isinstance(pnl_node.function, StatefulFunction):
                pnl_node.function.parameters.previous_value._set(value, context)
            # Do same for integrator_function of TransferMechanism if it is in integrator_mode
            if isinstance(pnl_node, TransferMechanism) and pnl_node.integrator_mode:
                pnl_node.integrator_function.parameters.previous_value._set(pytorch_node.integrator_previous_value,
                                                                            context)
        # Finally, update the output_values of the autodiff Composition by executing its output_CIM
        update_autodiff_all_output_values()

    def log_values(self):
        for node_wrapper in [n for n in self.wrapped_nodes if not isinstance(n, PytorchCompositionWrapper)]:
            node_wrapper.log_value()


class PytorchGRUMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for a GRU Node"""

    def _assign_pytorch_function(self, mechanism, device, context):
        self.function = PytorchGRUFunctionWrapper(mechanism.function, device, context)

        self.input_ports = [PytorchGRUFunctionWrapper(input_port.function, device, context)
                            for input_port in mechanism.input_ports]

    def execute(self, variable, context):
        """Execute Mechanism's _gen_pytorch version of function on variable.
        Enforce result to be 2d, and assign to self.output
        """
        def execute_function(function, variable, fct_has_mult_args=False):
            """Execute _gen_pytorch_fct on variable, enforce result to be 2d, and return it
            If fct_has_mult_args is True, treat each item in variable as an arg to the function
            If False, compute function for each item in variable and return results in a list
            """
            from psyneulink.core.components.functions.nonstateful.transformfunctions import TransformFunction
            if fct_has_mult_args:
                res = function(*variable)
            # variable is ragged
            elif isinstance(variable, list):
                res = [function(variable[i]) for i in range(len(variable))]
            else:
                res = function(variable)
            # TransformFunction can reduce output to single item from
            # multi-item input
            if isinstance(function._pnl_function, TransformFunction):
                res = res.unsqueeze(0)
            return res

        # If mechanism has an integrator_function and integrator_mode is True,
        #   execute it first and use result as input to the main function;
        #   assumes that if PyTorch node has been assigned an integrator_function then _mechanism has an integrator_mode
        if hasattr(self, 'integrator_function') and self._mechanism.parameters.integrator_mode._get(context):
            variable = execute_function(self.integrator_function,
                                        [self.integrator_previous_value, variable],
                                        fct_has_mult_args=True)
            # Keep track of previous value in Pytorch node for use in next forward pass
            self.integrator_previous_value = variable

        self.input = variable

        # Compute main function of mechanism and return result
        self.output = execute_function(self.function, variable)
        return self.output

    def log_value(self):
        # FIX: LOG HIDDEN STATE OF COMPOSITION MECHANISM
        if self._mechanism.parameters.value.log_condition != LogCondition.OFF:
            detached_value = self.output.detach().cpu().numpy()
            self._mechanism.output_port.parameters.value._set(detached_value, self._context)
            self._mechanism.parameters.value._set(detached_value, self._context)

    def execute(self, variable):
        # return torch.matmul(variable, self.matrix)
        return self.function(variable, self.matrix)

    def log_matrix(self):
        if self._projection.parameters.matrix.log_condition != LogCondition.OFF:
            detached_matrix = self.matrix.detach().cpu().numpy()
            self._projection.parameters.matrix._set(detached_matrix, context=self._context)
            self._projection.parameter_ports['matrix'].parameters.value._set(detached_matrix, context=self._context)

class PytorchGRUFunctionWrapper(PytorchFunctionWrapper):
    def __init__(self, function, device, context=None):
        self._pnl_function = function
        self.name = f"PytorchFunctionWrapper[GRU NODE]"
        self._context = context
        self.function = function

    def __repr__(self):
        return "PytorchWrapper for: " + self._pnl_function.__repr__()

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
