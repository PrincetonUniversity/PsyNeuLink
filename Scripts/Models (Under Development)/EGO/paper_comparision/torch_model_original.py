"""
Implementation of EGO Model for Study 2 of the paper `Giallanza et al. (2024)<https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00143/121081/Toward-the-Emergence-of-Intelligent-Control>`_

Reimplemntation from `https://github.com/thisisadax/ACSS-2024-EGO`_ with minimal changes for integration with PsyNeuLink.

ChangeSet: `./changeSets/Original.md`_


"""

import torch
import torch.nn as nn

import utils


class EMModule(nn.Module):
    """
    An Episodic Memory module that can be used as a sub-component of other models.

    The EM module is a key-value memory that stores a set of keys and values.
    When queried with a key, it returns a weighted sum of the values, where the weights
    are determined by the similarity between the query key and the stored keys.
    """

    def __init__(self, temperature, normalize_keys=True) -> None:
        super().__init__()
        self.state_keys = None
        self.context_keys = None
        self.values = None
        self.temperature = temperature
        self.normalize_keys = normalize_keys
        self.state_weight = nn.Parameter(torch.zeros(1))
        # Pointer to next write location
        self.index = 0

    def norm_key(self, key: torch.tensor) -> torch.tensor:
        """
        Normalize the provided key to unit length.

        Args:
            key: the key to normalize.
        """
        if self.normalize_keys:
            return key / key.norm(dim=-1, keepdim=True)
        else:
            return key

    def get_match_weights(self, state: torch.tensor, context: torch.tensor) -> torch.tensor:
        state = self.norm_key(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        _state_match_weights = torch.einsum('b a, c a -> c b', self.state_keys, state) / self.temperature

        context = self.norm_key(context)
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        _context_match_weights = torch.einsum('b a, c a -> c b', self.context_keys, context) / self.temperature
        return _state_match_weights + _context_match_weights

    def forward(self, state: torch.tensor, context: torch.tensor) -> torch.tensor:
        _match_weights = self.get_match_weights(state, context)
        matches = torch.einsum('a b, c a -> c b', self.values, utils.safe_softmax(_match_weights, dim=-1))
        return torch.clamp(matches, min=0, max=1)

    def prep(self, state_key, context_key, value):
        state_key, context_key = self.norm_key(state_key), self.norm_key(context_key)
        if self.state_keys is None:
            self.state_keys = state_key
        else:
            self.state_keys = torch.cat((self.state_keys, state_key), dim=0)
        if self.context_keys is None:
            self.context_keys = context_key
        else:
            self.context_keys = torch.cat((self.context_keys, context_key), dim=0)
        if self.values is None:
            self.values = value
        else:
            self.values = torch.cat((self.values, value), dim=0)

    def write(self, state_key, context_key, value):
        state_key, context_key = self.norm_key(state_key), self.norm_key(context_key)
        if self.state_keys is None:
            self.state_keys = state_key
        else:
            self.state_keys[self.index] = state_key
        if self.context_keys is None:
            self.context_keys = context_key
        else:
            self.context_keys[self.index] = context_key
        if self.values is None:
            self.values = value
        else:
            self.values[self.index] = value
        self.index += 1


class RecurrentContextModule(nn.Module):
    """
    An Recurrent Neural Network module based on an architecture similar to the minimally gated recurrent unit.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs) -> None:
        super().__init__()
        self.state_to_hidden = nn.Linear(n_inputs, n_hidden)
        self.hidden_to_hidden = nn.Linear(n_hidden, n_hidden)
        self.state_to_hidden_wt = nn.Linear(n_inputs, n_hidden)
        self.hidden_to_hidden_wt = nn.Linear(n_hidden, n_hidden)
        self.hidden_to_context = nn.Linear(n_hidden, n_outputs)
        self.n_hidden_units = n_hidden
        self.hidden_state = torch.zeros((self.n_hidden_units,), dtype=torch.float)
        self.update_hidden_state = True

    def forward(self, x: torch.tensor) -> torch.tensor:
        h_prev = self.hidden_state
        h_update = torch.tanh(self.state_to_hidden(x) + self.hidden_to_hidden(h_prev))
        h_weight = torch.sigmoid(self.state_to_hidden_wt(x) + self.hidden_to_hidden_wt(h_prev))
        h_new = h_weight * h_prev + (1 - h_weight) * h_update
        if self.update_hidden_state:
            self.hidden_state = h_new.detach().clone()
        return self.hidden_to_context(h_new)


def prep_em(em, n, l, fill=.001):
    for name, p in em.named_parameters():
        if 'context_in' not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    for _ in range(n):
        em.prep(torch.tensor([[fill] * l], dtype=torch.float),
                torch.tensor([[fill] * l], dtype=torch.float),
                torch.tensor([[fill] * l], dtype=torch.float))
    return em


def prep_recurrent_network(rnet, state_d, persistance=-0.6):
    with torch.no_grad():
        rnet.state_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.state_to_hidden.bias.zero_()
        rnet.hidden_to_hidden.weight.zero_()
        rnet.hidden_to_hidden.bias.zero_()
        rnet.state_to_hidden_wt.weight.zero_()
        rnet.state_to_hidden_wt.bias.copy_(
            torch.ones((len(rnet.state_to_hidden_wt.bias),), dtype=torch.float) * persistance)
        rnet.hidden_to_hidden_wt.weight.zero_()
        rnet.hidden_to_hidden_wt.bias.zero_()
        # Set hidden to context weights as an identity matrix.
        rnet.hidden_to_context.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.hidden_to_context.bias.zero_()

    # Set requires_grad to True for hidden_to_context.weight before freezing other parameters
    rnet.hidden_to_context.weight.requires_grad = True
    rnet.hidden_to_context.bias.requires_grad = True

    # Freeze recurrent weights to stabilize training
    for name, p in rnet.named_parameters():
        if 'hidden_to_context' not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    return rnet


def gen_model(params):
    context_module = RecurrentContextModule(params.state_d, params.state_d, params.context_d)
    em_module = EMModule(params.temperature)
    context_module = prep_recurrent_network(context_module, params.state_d, params.persistance)
    em_module = prep_em(em_module, params.memory_len, params.context_d, params.memory_init)
    return context_module, em_module
