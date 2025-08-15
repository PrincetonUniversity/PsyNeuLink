import torch
import torch.nn as nn

torch.set_printoptions(precision=8)


def prep_EM(em, n, l, fill=.001):
    for _ in range(n):
        em.prep(torch.tensor([[fill] * l], dtype=torch.float),
                torch.tensor([[fill] * l], dtype=torch.float),
                torch.tensor([[fill] * l], dtype=torch.float))
    return em


def safe_softmax(t, threshold=0.01, **kwargs):
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


class EMModule(nn.Module):
    """
    An Epsiodic Memory module that can be used as a sub-component of other models.

    The EM module is a key-value memory that stores a set of keys and values.
    When queried with a key, it returns a weighted sum of the values, where the weights
    are determined by the similarity between the query key and the stored keys.
    """

    def __init__(self, temperature, softmax_threshold, normalize_keys=False) -> None:
        super().__init__()
        self.index = 0
        self.state_keys = None
        self.context_keys = None
        self.values = None
        self.encode_context = True

        self.temperature = temperature
        self.softmax_threshold = softmax_threshold
        self.normalize_keys = normalize_keys  # normalize_keys

        self.state_weight = nn.Parameter(torch.zeros(1))

        self.state_match_weights_ = None
        self.context_match_weights_ = None

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
        if not self.encode_context:
            state = torch.cat([state, context], axis=-1)
        state = self.norm_key(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        self.state_match_weights_ = torch.einsum('b a, c a -> c b', self.state_keys, state) / self.temperature

        context = self.norm_key(context)
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        self.context_match_weights_ = torch.einsum('b a, c a -> c b', self.context_keys, context) / self.temperature
        return self.state_match_weights_ + self.context_match_weights_

    def forward(self, state: torch.tensor, context: torch.tensor) -> torch.tensor:
        self.match_weights_ = self.get_match_weights(state, context)
        matches = torch.einsum('a b, c a -> c b', self.values,
                               safe_softmax(self.match_weights_, self.softmax_threshold, dim=-1))

        return torch.clamp(matches, min=0, max=1)

    def prep(self, state_key, context_key, value):
        if not self.encode_context:
            state_key = torch.cat([state_key, context_key], dim=-1)
        state_key, context_key = self.norm_key(state_key), self.norm_key(context_key)
        if self.state_keys is None:
            self.state_keys = state_key
        else:
            self.state_keys = torch.cat((self.state_keys, state_key), dim=0)
        if self.encode_context:
            if self.context_keys is None:
                self.context_keys = context_key
            else:
                self.context_keys = torch.cat((self.context_keys, context_key), dim=0)
        if self.values is None:
            self.values = value
        else:
            self.values = torch.cat((self.values, value), dim=0)

    def write(self, state_key, context_key, value):

        if not self.encode_context:
            state_key = torch.cat([state_key, context_key], dim=-1)
        state_key, context_key = self.norm_key(state_key), self.norm_key(context_key)
        if self.state_keys is None:
            self.state_keys = state_key
        else:
            self.state_keys[self.index] = state_key
        if self.encode_context:
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

    def __init__(self, n_inputs, n_hidden, n_outputs, integration_rate=.5) -> None:
        super().__init__()
        self.integration_rate = integration_rate
        self.state_to_hidden = nn.Linear(n_inputs, n_hidden)
        self.hidden_to_hidden = nn.Linear(n_hidden, n_hidden)
        self.hidden_to_context = nn.Linear(n_hidden, n_outputs)
        self.n_hidden_units = n_hidden
        self.hidden_state = torch.zeros((self.n_hidden_units,), dtype=torch.float)
        self.update_hidden_state = True

    def forward(self, x: torch.tensor) -> torch.tensor:
        h_prev = self.hidden_state
        h_new = self.integration_rate * self.state_to_hidden(x) + (1 - self.integration_rate) * self.hidden_to_hidden(
            h_prev)

        if self.update_hidden_state:
            self.hidden_state = h_new.detach().clone()
        return self.hidden_to_context(torch.tanh(h_new))


def prep_recurrent_network(rnet, state_d):
    with torch.no_grad():
        rnet.state_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.state_to_hidden.bias.zero_()

        rnet.hidden_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.hidden_to_hidden.bias.zero_()
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


def gen_model(params, len_memory=2):
    context_module = RecurrentContextModule(
        params['state_d'],
        params['state_d'],
        params['context_d'],
        params['integration_rate']
    )
    em_module = EMModule(
        params['temperature'],
        params['softmax_threshold']
    )
    context_module = prep_recurrent_network(context_module, params['state_d'])
    em_module = prep_EM(em_module, len_memory, params['state_d'], params['memory_init'])
    return context_module, em_module
