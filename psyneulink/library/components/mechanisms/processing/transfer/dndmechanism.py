# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND INTEGRATION_RATE ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

# ****************************************  DNDMechanism ***************************************************************

"""the DND class
notes:
- memory is a row vector
"""

import torch
import torch.nn.functional as F

from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.globals.keywords import NAME, SIZE
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set

# constants
ALL_KERNELS = ['cosine', 'l1', 'l2']
ALL_POLICIES = ['1NN']


class DND(ProcessingMechanism):
    """The differentiable neural dictionary (DND) class. This enables episodic recall in a neural network.

    Parameters
    ----------
    dict_len : int
        the maximial len of the dictionary
    memory_dim : int
        the dim or len of memory i, we assume memory_i is a row vector
    kernel : str
        the metric for memory search

    Attributes
    ----------
    encoding_off : bool
        if True, stop forming memories
    retrieval_off : type
        if True, stop retrieving memories
    reset_memory : func;
        if called, clear the dictionary
    check_config : func
        check the class config

    """
    def __init__(self,
                 default_variable=None,
                 key_size=1,
                 value_size=1,
                 dict_len=1,
                 kernel='1NN',
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # params
        self.dict_len = dict_len
        self.kernel = kernel
        self.memory_dim = value_size
        # dynamic state
        self.encoding_off = False
        self.retrieval_off = False
        # allocate space for memories
        self.reset_memory()

        self.input_states = [{NAME:'KEY INPUT', SIZE:key_size},
                             {NAME:'VALUE INPUT', SIZE:value_size}]

        super().__init__(default_variable=default_variable,
                         params=params,
                         name=name,
                         prefs=prefs)

    def function(self, var):
        super()._execute(self, variable=None, execution_id=None, runtime_params=None, context=None, **kwargs):

    # def _execute(self,
    #     variable=None,
    #     execution_id=None,
    #     runtime_params=None,
    #     context=None
    # ):
    #     pass
    #
    def reset_memory(self):
        self.keys = []
        self.vals = []

    def _validate_params(self, request_set, target_set=None, context=None):
        assert self.dict_len > 0
        assert self.kernel in ALL_KERNELS

    def inject_memories(self, input_keys, input_vals):
        """Inject pre-defined keys and values

        Parameters
        ----------
        input_keys : list
            a list of memory keys
        input_vals : list
            a list of memory content
        """
        assert len(input_keys) == len(input_vals)
        for k, v in zip(input_keys, input_vals):
            self.save_memory(k, v)

    def save_memory(self, memory_key, memory_val):
        """Save an episodic memory to the dictionary

        Parameters
        ----------
        memory_key : a row vector
            a DND key, used to for memory search
        memory_val : a row vector
            a DND value, representing the memory content
        """
        if self.encoding_off:
            return
        # add new memory to the the dictionary
        # get data is necessary for gradient reason
        self.keys.append(memory_key.data.view(1, -1))
        self.vals.append(memory_val.data.view(1, -1))
        # remove the oldest memory, if overflow
        if len(self.keys) > self.dict_len:
            self.keys.pop(0)
            self.vals.pop(0)

    def get_memory(self, query_key):
        """Perform a 1-NN search over dnd

        Parameters
        ----------
        query_key : a row vector
            a DND key, used to for memory search

        Returns
        -------
        a row vector
            a DND value, representing the memory content

        """
        # if no memory, return the zero vector
        n_memories = len(self.keys)
        if n_memories == 0 or self.retrieval_off:
            return empty_memory(self.memory_dim)
        # compute similarity(query, memory_i ), for all i
        similarities = compute_similarities(query_key, self.keys, self.kernel)
        # get the best-match memory
        best_memory_val = self._get_memory(similarities)
        return best_memory_val

    def _get_memory(self, similarities, policy='1NN'):
        """get the episodic memory according to some policy
        e.g. if the policy is 1nn, return the best matching memory
        e.g. the policy can be based on the rational model

        Parameters
        ----------
        similarities : a vector of len #memories
            the similarity between query vs. key_i, for all i
        policy : str
            the retrieval policy

        Returns
        -------
        a row vector
            a DND value, representing the memory content

        """
        best_memory_val = None
        if policy is '1NN':
            best_memory_id = torch.argmax(similarities)
            best_memory_val = self.vals[best_memory_id]
        else:
            assert False, 'ERROR IN DND'
        return best_memory_val


"""helpers"""


def compute_similarities(query_key, key_list, metric):
    """Compute the similarity between query vs. key_i for all i
        i.e. compute q M, w/ q: 1 x key_dim, M: key_dim x #keys

    Parameters
    ----------
    query_key : a row vector
        Description of parameter `query_key`.
    key_list : list
        Description of parameter `key_list`.
    metric : str
        Description of parameter `metric`.

    Returns
    -------
    a row vector w/ len #memories
        the similarity between query vs. key_i, for all i

    """
    # query_key = query_key.data
    # reshape query to 1 x key_dim
    q = query_key.data.view(1, -1)
    # reshape memory keys to #keys x key_dim
    M = torch.stack(key_list, dim=1).view(len(key_list), -1)
    # compute similarities
    if metric is 'cosine':
        similarities = F.cosine_similarity(q, M)
    elif metric is 'l1':
        similarities = - F.pairwise_distance(q, M, p=1)
    elif metric is 'l2':
        similarities = - F.pairwise_distance(q, M, p=2)
    else:
        assert False, 'ERROR IN DND'
    return similarities


def empty_memory(memory_dim):
    """Get a empty memory, assuming the memory is a row vector
    """
    return torch.zeros(1, memory_dim)
