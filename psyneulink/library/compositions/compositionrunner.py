# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* AutodiffComposition *************************************************
import random
import numpy as np
import collections.abc
import inspect

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.utilities import NodeRole
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.globals.keywords import TARGET_MECHANISM, COMPARATOR_MECHANISM, LEARNING_MECHANISM, TRAINING_SET

__all__ = ["CompositionRunner"]

def inf_yield_none():
    while True:
        yield None

def _chunk_inputs(inputs: dict, num_trials: int, chunksize: int = 1, randomize: bool = True):
    """
    Chunks input dict into pieces where each chunk is a dict with values of length chunksize (or for the last chunk, the remainder)
    """
    chunks = []
    indices = list(range(0, num_trials))
    if randomize:
        random.shuffle(indices)

    for i in range(0, num_trials, chunksize):
        curr_indices = indices[i:i + chunksize]
        chunk = {}
        for k, v in inputs.items():
            chunk[k] = [v[i % len(v)] for i in curr_indices]
        chunks.append((chunk, curr_indices))
    return chunks

def _recursive_update(d, u):
    """
    Recursively calls update on dictionaries, which prevents deletion of keys
    """
    for key, val in u.items():
        if isinstance(val, collections.abc.Mapping):
            d[key] = _recursive_update(d.get(key, {}), val)
        else:
            d[key] = val
    return d
class CompositionRunner():

    def __init__(self, compostion: Composition):
        self._composition = compostion

    def _parse_stim_inputs(self, inputs: dict, targets: dict):
        """
        Converts inputs and targets to a standardized form

        Returns
        ---------
        Dict mapping mechanisms to values (with TargetMechanisms inferred if needed)
        """
        # 1) Convert from key-value representation of values into separated representation
        if 'targets' in inputs:
            targets = inputs['targets'].copy()

        if 'inputs' in inputs:
            inputs = inputs['inputs'].copy()

        # 2) Convert output node keys -> target node keys (learning always needs target nodes!)
        if targets is not None:
            targets = self._infer_target_nodes(targets)
            inputs = _recursive_update(inputs, targets)

        return inputs

    def _infer_target_nodes(self, targets: dict):
        """
        Maps targets onto target mechanisms (as needed by learning)

        Returns
        ---------
        A dict mapping TargetMechanisms -> target values
        """
        ret = {}
        for node, values in targets.items():
            if NodeRole.TARGET not in self._composition.get_roles_by_node(node) and NodeRole.LEARNING not in self._composition.get_roles_by_node(node):
                node_efferent_mechanisms = [x.receiver.owner for x in node.efferents]
                comparators = [x for x in node_efferent_mechanisms if (isinstance(x, ComparatorMechanism) and NodeRole.LEARNING in self._composition.get_roles_by_node(x))]
                comparator_afferent_mechanisms = [x.sender.owner for c in comparators for x in c.afferents]
                target_nodes = [t for t in comparator_afferent_mechanisms if (NodeRole.TARGET in self._composition.get_roles_by_node(t) and NodeRole.LEARNING in self._composition.get_roles_by_node(t))]

                if len(target_nodes) != 1:
                    # Invalid specification! Either we have no valid target nodes, or there is ambiguity in which target node to choose
                    raise Exception(f"Unable to infer learning target node from output node {node}!")

                ret[target_nodes[0]] = values
            else:
                ret[node] = values
        return ret

    def _get_loss(self):
        """
        Returns a value that is the sum of all the losses from the last iteration
        """
        total_loss = 0
        for terminal_sequence in self._composition._terminal_backprop_sequences.values():
            comparator = terminal_sequence[COMPARATOR_MECHANISM]
            total_loss += comparator.value[0][0]

        return total_loss

    def run_learning(self,
                     inputs: dict,
                     targets: dict = None,
                     num_trials: int = None,
                     epochs: int = 1,
                     minibatch_size: int = 1,
                     patience: int = None,
                     min_delta: int = 0,
                     randomize_minibatches: bool = True,
                     call_before_minibatch = None,
                     call_after_minibatch = None,
                     context=None):
        """
        Runs the composition repeatedly with the specified parameters

        Returns
        ---------
        Outputs from the final execution
        """

        # Handle function and generator inputs
        if callable(inputs):
            inputs = inputs()

        if isinstance(inputs, dict):
            inputs = [inputs]

        if callable(targets):
            targets = targets()

        if isinstance(targets, dict):
            targets = [targets]
        elif targets is None:
            targets = inf_yield_none()

        if callable(epochs):
            epochs = epochs()

        if not isinstance(epochs, list) and not isinstance(epochs, tuple):
            epochs = [epochs]
        elif epochs is None:
            epochs = inf_yield_none()


        for stim_input, stim_target, stim_epoch in zip(inputs, targets, epochs):
            if 'epochs' in stim_input:
                stim_epoch = stim_input['epochs']

            stim_input = self._parse_stim_inputs(stim_input, stim_target)

            if num_trials is None:
                num_trials = len(list(stim_input.values())[0])

            if minibatch_size == TRAINING_SET:
                minibatch_size = num_trials

            if patience is not None:
                early_stopper = EarlyStopping(min_delta=min_delta, patience=patience)

            skip_initialization = False
            for curr_epoch in range(stim_epoch):
                results = []
                for minibatch, indices in _chunk_inputs(stim_input, num_trials, minibatch_size, randomize_minibatches):
                    if call_before_minibatch is not None:
                        call_before_minibatch()

                    minibatch_results = self._composition.run(inputs=minibatch, skip_initialization=skip_initialization, context=context, skip_analyze_graph=skip_initialization)
                    skip_initialization = True
                    results.extend(minibatch_results)

                    if call_after_minibatch is not None:
                        call_after_minibatch()
                epoch_loss = self._get_loss()
                if (patience is not None and early_stopper.step(epoch_loss)) or curr_epoch == stim_epoch - 1:
                    break
        
        return results

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
