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

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.utilities import NodeRole
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.globals.keywords import TARGET_MECHANISM, COMPARATOR_MECHANISM, LEARNING_MECHANISM

__all__ = ["CompositionRunner"]
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
            chunk[k] = [v[i] for i in curr_indices]
        chunks.append((chunk, curr_indices))
    return chunks

class CompositionRunner():

    def __init__(self, compostion: Composition):
        self._composition = compostion
    
    def _parse_inputs(self, inputs: dict, targets: dict):
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
            inputs.update(targets)

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
        for output_node, terminal_sequence in self._composition._terminal_backprop_sequences.items():
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
                     context=None):
        """
        Runs the composition repeatedly with the specified parameters

        Returns
        ---------
        Outputs from the final execution
        """

        if 'epochs' in inputs:
            epochs = inputs['epochs']

        inputs = self._parse_inputs(inputs, targets)

        if num_trials is None:
            num_trials = len(list(inputs.values())[0])

        results = [None] * num_trials
        if patience is not None:
            early_stopper = EarlyStopping(min_delta=min_delta, patience=patience)

        skip_initialization = False
        for curr_epoch in range(epochs):
            for minibatch, indices in _chunk_inputs(inputs, num_trials, minibatch_size, randomize_minibatches):
                minibatch_results = self._composition.run(inputs=minibatch, learning_mode=True, skip_initialization=skip_initialization, skip_analyze_graph=skip_initialization, context=context)
                skip_initialization = True
                for i, j in enumerate(indices):
                    # Reorder minibatch to match up with original order
                    results[j] = minibatch_results[i]
            epoch_loss = self._get_loss()
            if (patience is not None and early_stopper.step(epoch_loss)) or curr_epoch == epochs - 1:
                # return highest index result
                return results[-1]

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
