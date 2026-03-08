import torch
import torch.nn as nn


def safe_softmax(t, threshold=0.01):
    """
    Softmax function that always sums to 1 or less. Handles occasional numerical errors in torch's softmax.
    Nullifies values below the given threshold.
    """
    v = t

    # Apply mask: only include values greater than mask_threshold
    if threshold is not None:
        v = torch.where(abs(t) > threshold, v, torch.tensor(-torch.inf, device=t.device))

    # Shift by the global max to avoid extreme values
    if torch.any(v != -torch.inf):
        v = v - torch.max(v)

    # Exponential
    v = torch.exp(v)

    # Normalize (to sum to 1)
    if not v.any():
        return v
    else:
        return v / torch.sum(v)


def normalize(x):
    nrm = x.norm(dim=-1, keepdim=True)
    if nrm == 0:
        return x
    return x / nrm


class EMModule(nn.Module):
    """
    An Episodic Memory module that can be used as a sub-component of other models.

    The EM module is a key-value memory that stores a set of keys and values.
    When queried with a key, it returns a weighted sum of the values, where the weights
    are determined by the similarity between the query key and the stored keys.
    """

    def __init__(self,
                 state_dim,
                 temperature,
                 softmax_threshold,
                 memory_fill=.001,
                 memory_fill_n=None) -> None:
        super().__init__()

        self.state_keys = None
        self.context_keys = None
        self.values = None

        self.temperature = temperature
        self.softmax_threshold = softmax_threshold

        self.index = 0

        self.initialize_memories(memory_fill_n, state_dim, memory_fill)

    def get_match_weights(self, state: torch.tensor, context: torch.tensor) -> torch.tensor:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state_match_weights_ = torch.einsum('b a, c a -> c b', self.state_keys, state) / self.temperature
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        context_match_weights_ = torch.einsum('b a, c a -> c b', self.context_keys, context) / self.temperature
        return state_match_weights_ + context_match_weights_

    def forward(self, state: torch.tensor, context: torch.tensor) -> torch.tensor:
        match_weights_ = self.get_match_weights(state, context)
        matches = torch.einsum('a b, c a -> c b', self.values,
                               safe_softmax(match_weights_, self.softmax_threshold))

        return torch.clamp(matches, min=0, max=1)

    def initialize_memories(self, n, l, fill=.001):
        self.state_keys = torch.full((n, l), fill)
        self.context_keys = torch.full((n, l), fill)
        self.values = torch.full((n, l), fill)
        self.index = 0

    def write(self, state_key, context_key, value):
        self.state_keys[self.index] = state_key
        self.context_keys[self.index] = context_key
        self.values[self.index] = value
        self.index += 1


class RecurrentContextModule(nn.Module):
    """
    A Recurrent Neural Network module based on an architecture similar to the minimally gated recurrent unit.
    """

    def __init__(self, state_dim, integration_rate=.5) -> None:
        super().__init__()
        self.integration_rate = integration_rate
        self.hidden_state = torch.zeros((state_dim,))

    def forward(self, x: torch.tensor) -> torch.tensor:
        h_new = self.integration_rate * x + (1 - self.integration_rate) * self.hidden_state
        self.hidden_state = h_new.detach().clone()
        return h_new


class ContextMapping(nn.Module):
    """
    A Recurrent Neural Network module based on an architecture similar to the minimally gated recurrent unit.
    """

    def __init__(self, state_dim) -> None:
        super().__init__()
        self.in_to_out = nn.Linear(state_dim, state_dim, bias=False)
        with torch.no_grad():
            self.in_to_out.weight.copy_(torch.eye(state_dim, dtype=torch.float))
            #self.in_to_out.bias.zero_()
        self.in_to_out.weight.requires_grad = True
        #self.in_to_out.bias.requires_grad = True

    def forward(self, x: torch.tensor) -> torch.tensor:
        return normalize(self.in_to_out(x))


def gen_model(params, len_memory=2):
    context_module = RecurrentContextModule(
        params['state_d'],
        params['integration_rate']
    )
    context_mapping = ContextMapping(
        params['state_d'],
    )
    em_module = EMModule(
        params['state_d'],
        params['temperature'],
        params['softmax_threshold'],
        memory_fill=params['memory_fill'],
        memory_fill_n=len_memory
    )
    return context_module, context_mapping, em_module
