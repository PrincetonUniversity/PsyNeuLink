from psyneulink.core.components.functions.function import Linear, Logistic
try:
    import torch
    from torch import nn
    torch_available = True
except ImportError:
    torch_available = False

import numpy as np

__all__ = ['PytorchModelCreator']
# Class that is called to create pytorch representations of autodiff compositions based on their processing graphs.
# Called to do so when the composition is run for the first time.

# Note on notation: the "nodes" that are constantly referred to are vertices of the composition's processing
# graph. For general compositions, the component a node represents can be a mechanism or a nested composition,
# but for autodiff compositions, nodes always represent mechanisms. "Nodes" can be thought of as
# (but are not literally) mechanisms.


class PytorchModelCreator(torch.nn.Module):

    # sets up parameters of model & the information required for forward computation
    def __init__(self, processing_graph, param_init_from_pnl, ordered_execution_sets, execution_id=None):

        if not torch_available:
            raise Exception('Pytorch python module (torch) is not installed. Please install it with '
                            '`pip install torch` or `pip3 install torch`')

        super(PytorchModelCreator, self).__init__()

        self.ordered_execution_sets = ordered_execution_sets  # saved for use in the forward method
        self.component_to_forward_info = {}  # dict mapping PNL nodes to their forward computation information
        self.projections_to_pytorch_weights = {}  # dict mapping PNL projections to Pytorch weights
        self.mechanisms_to_pytorch_biases = {}  # dict mapping PNL mechanisms to Pytorch biases
        self.params = nn.ParameterList()  # list that Pytorch optimizers will use to keep track of parameters

        for i in range(len(self.ordered_execution_sets)):
            for component in self.ordered_execution_sets[i]:

                value = None  # the node's (its mechanism's) value
                biases = None  # the node's bias parameters
                function = self.function_creator(component, execution_id)  # the node's function
                afferents = {}  # dict for keeping track of afferent nodes and their connecting weights

                # if `node` is not an origin node (origin nodes don't have biases or afferent connections)
                if i != 0:

                    # if not copying parameters from psyneulink, set up pytorch biases for node
                    if not param_init_from_pnl:
                        biases = nn.Parameter(torch.zeros(len(component.input_states[0].parameters.value.get(execution_id))).double())
                        self.params.append(biases)
                        self.mechanisms_to_pytorch_biases[component] = biases

                    # iterate over incoming projections and set up pytorch weights for them
                    for k in range(len(component.path_afferents)):

                        # get projection, sender node for projection
                        mapping_proj = component.path_afferents[k]
                        input_component = mapping_proj.sender.owner
                        input_node = processing_graph.comp_to_vertex[input_component]

                        # set up pytorch weights that correspond to projection. If copying params from psyneulink,
                        # copy weight values from projection. Otherwise, use random values.
                        if param_init_from_pnl:
                            weights = nn.Parameter(torch.tensor(mapping_proj.parameters.matrix.get(execution_id).copy()).double())
                        else:
                            weights = nn.Parameter(torch.rand(np.shape(mapping_proj.parameters.matrix.get(execution_id))).double())
                        afferents[input_node] = weights
                        self.params.append(weights)
                        self.projections_to_pytorch_weights[mapping_proj] = weights

                node_forward_info = [value, biases, function, afferents]

                self.component_to_forward_info[component] = node_forward_info

    # performs forward computation for the model
    def forward(self, inputs, execution_id=None):

        outputs = {}  # dict for storing values of terminal (output) nodes

        for i in range(len(self.ordered_execution_sets)):
            for component in self.ordered_execution_sets[i]:

                # get forward computation info for current component
                biases = self.component_to_forward_info[component][1]
                function = self.component_to_forward_info[component][2]
                afferents = self.component_to_forward_info[component][3]

                # forward computation if we have origin node
                if i == 0:
                    value = function(inputs[component])

                # forward computation if we do not have origin node
                else:
                    value = torch.zeros(len(component.input_states[0].parameters.value.get(execution_id))).double()
                    for input_node, weights in afferents.items():
                        value += torch.matmul(self.component_to_forward_info[input_node.component][0], weights)
                    if biases is not None:
                        value = value + biases
                    value = function(value)

                # store the current value of the node
                self.component_to_forward_info[component][0] = value

                # save value in output list if we're at a node in the last execution set
                if i == len(self.ordered_execution_sets) - 1:
                    outputs[component] = value

        return outputs

    # helper method that identifies the type of function used by a node, gets the function
    # parameters and uses them to create a function object representing the function, then returns it
    def function_creator(self, node, execution_id=None):
        def get_fct_param_value(param_name):
            return float(node.function_object.get_current_function_param(param_name, execution_id))

        if isinstance(node.function_object, Linear):
            slope = get_fct_param_value('slope')
            intercept = get_fct_param_value('intercept')
            return lambda x: x * slope + intercept

        elif isinstance(node.function_object, Logistic):
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            offset = get_fct_param_value('offset')
            return lambda x: 1 / (1 + torch.exp(-gain * (x - bias) + offset))

        else:  # if we have relu function (the only other kind of function allowed by the autodiff composition)
            gain = get_fct_param_value('gain')
            bias = get_fct_param_value('bias')
            leak = get_fct_param_value('leak')
            return lambda x: (torch.max(input=(x - bias), other=torch.tensor([0]).double()) * gain +
                              torch.min(input=(x - bias), other=torch.tensor([0]).double()) * leak)

    # returns dict mapping psyneulink projections to corresponding pytorch weights. Pytorch weights are copied
    # over from tensors inside Pytorch's Parameter data type to numpy arrays (and thus copied to a different
    # memory location). This keeps the weights - and Pytorch in general - away from the user
    def get_weights_for_projections(self):
        weights_in_numpy = {}
        for projection, weights in self.projections_to_pytorch_weights.items():
            weights_in_numpy[projection] = weights.detach().numpy().copy()
        return weights_in_numpy

    # returns dict mapping psyneulink mechanisms to corresponding pytorch biases, the same way as the above function.
    # If composition is initialized with "param_init_from_PNL" set to true, then no biases are created in Pytorch,
    # and when called, this function returns an empty list.
    def get_biases_for_mechanisms(self):
        biases_in_numpy = {}
        for mechanism, biases in self.mechanisms_to_pytorch_biases.items():
            biases_in_numpy[mechanism] = biases.detach().numpy().copy()
        return biases_in_numpy
