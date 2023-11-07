# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* AutodiffComposition *************************************************

import numpy as np

from psyneulink.core.llvm import ExecutionMode
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.compositions.report import Report, ReportProgress, ReportDevices, LEARN_REPORT, PROGRESS_REPORT
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import LearningMechanism
from psyneulink.core.globals.keywords import OBJECTIVE_MECHANISM, TRAINING_SET
from inspect import isgeneratorfunction

__all__ = ["CompositionRunner"]

def inf_yield_val(val=None):
    while True:
        yield val

class CompositionRunner():

    def __init__(self, compostion: Composition):
        self._composition = compostion

    def _calculate_loss(self, num_trials:int, execution_mode:ExecutionMode, context):
        """
        Returns a value that is the sum of all the losses from the last iteration
        """
        from psyneulink.library.compositions import AutodiffComposition

        if isinstance(self._composition, AutodiffComposition):
            return self._composition._get_total_loss(num_trials, context)

        total_loss = 0
        for terminal_sequence in self._composition._terminal_backprop_sequences.values():
            comparator = terminal_sequence[OBJECTIVE_MECHANISM]
            total_loss += comparator.value[0][0]
        return total_loss

    def _batch_inputs(self,
                      inputs: dict,
                      epochs: int,
                      num_trials: int,
                      batch_size: int = 1,
                      randomize: bool = True,
                      call_before_minibatch=None,
                      call_after_minibatch=None,
                      early_stopper=None,
                      execution_mode:ExecutionMode=ExecutionMode.Python,
                      context=None):
        """
        Chunks input dict into pieces where each chunk is a dict with values of length batch_size
        (or for the last chunk, the remainder)
        """

        assert early_stopper is None or not self._is_llvm_mode, "Early stopper doesn't work in compiled mode"
        assert call_before_minibatch is None or not self._is_llvm_mode, "minibatch calls don't work in compiled mode"
        assert call_after_minibatch is None or not self._is_llvm_mode, "minibatch calls don't work in compiled mode"

        #This is a generator for performance reasons,
        #    since we don't want to copy any data (especially for very large inputs or epoch counts!)
        for epoch in range(epochs):
            indices = list(range(0, num_trials))
            if randomize:
                np.random.shuffle(indices)
            for i in range(0, num_trials, batch_size):
                if call_before_minibatch:
                    call_before_minibatch()
                curr_indices = indices[i:i + batch_size]
                for idx in curr_indices:
                    chunk = {}
                    for k, v in inputs.items():
                        chunk[k] = v[idx % len(v)]
                    yield chunk
                if call_after_minibatch:
                    call_after_minibatch()

                # Update weights if in PyTorch execution_mode;
                #  handled by Composition.execute in Python mode and in compiled version in LLVM mode
                if execution_mode is ExecutionMode.PyTorch:
                    self._composition._update_learning_parameters(context)

            # Compiled mode does not need more identical inputs.
            # number_of_runs will be set appropriately to cycle over the set
            if self._is_llvm_mode and not randomize:
                return
            if (not self._is_llvm_mode and early_stopper is not None
                    and early_stopper.step(self._calculate_loss(num_trials, execution_mode, context))):
                # end early if patience exceeded
                pass

    def _batch_function_inputs(self,
                               inputs: dict,
                               epochs: int,
                               num_trials: int,
                               batch_size: int = 1,
                               call_before_minibatch=None,
                               call_after_minibatch=None,
                               early_stopper=None,
                               execution_mode:ExecutionMode=ExecutionMode.Python,
                               context=None):

        assert early_stopper is None or not self._is_llvm_mode, "Early stopper doesn't work in compiled mode"
        assert call_before_minibatch is None or not self._is_llvm_mode, "minibatch calls don't work in compiled mode"
        assert call_after_minibatch is None or not self._is_llvm_mode, "minibatch calls don't work in compiled mode"

        for epoch in range(epochs):
            for i in range(0, num_trials, batch_size):
                batch_ran = False

                if call_before_minibatch:
                    call_before_minibatch()

                for idx in range(i, i + batch_size):
                    try:
                        trial_input, _ = self._composition._parse_learning_spec(inputs=inputs(idx),
                                                                                targets=None,
                                                                                execution_mode=execution_mode,
                                                                                context=context)
                    except:
                        break
                    if trial_input is None:
                        break
                    batch_ran = True
                    yield trial_input

                if batch_ran:
                    if call_after_minibatch:
                        call_after_minibatch()

                    # Update weights if in PyTorch execution_mode;
                    #  handled by Composition.execute in Python mode and in compiled version in LLVM mode
                    if execution_mode is ExecutionMode.PyTorch:
                        self._composition._update_learning_parameters(context)
                else:
                    break

            if (not self._is_llvm_mode
                    and early_stopper is not None
                    and early_stopper.step(self._calculate_loss(num_trials, execution_mode, context))):
                # end early if patience exceeded
                pass

    def run_learning(self,
                     inputs: dict,
                     targets: dict = None,
                     num_trials: int = None,
                     epochs: int = 1,
                     learning_rate = None,
                     minibatch_size: int = 1,
                     patience: int = None,
                     min_delta: int = 0,
                     randomize_minibatches: bool = True,
                     call_before_minibatch = None,
                     call_after_minibatch = None,
                     context=None,
                     execution_mode:ExecutionMode = ExecutionMode.Python,
                     **kwargs):
        """
        Runs the composition repeatedly with the specified parameters.

        Returns
        ---------
        Outputs from the final execution
        """

        if not (execution_mode & ExecutionMode.COMPILED):
            self._is_llvm_mode = False
        else:
            self._is_llvm_mode = True

        if execution_mode is ExecutionMode.Python and learning_rate is not None:
            # User learning_rate specified in call to learn, so use that by passing it in runtime_params,
            #   excluding any LearningMechanisms for which learning_rate has been individually specified
            runtime_params = {learning_mechanism:{'learning_rate':learning_rate}
                              for learning_mechanism in self._composition.nodes
                              if isinstance(learning_mechanism, LearningMechanism) and
                              learning_mechanism.parameters.learning_rate.get() == # If learning_rate != default
                              learning_mechanism.defaults.learning_rate}           # it was individually specified
            if 'runtime_params' in kwargs:
                kwargs['runtime_params'].update(runtime_params)
            else:
                kwargs['runtime_params'] = runtime_params
        else:
            # This is used by local learning-related methods to override the default learning_rate set at construction.
            self._composition._runtime_learning_rate = learning_rate

        # Handle function and generator inputs
        if isgeneratorfunction(inputs):
            inputs = inputs()

        if isinstance(inputs, dict) or callable(inputs):
            inputs = [inputs]

        if isgeneratorfunction(targets):
            targets = targets()

        if isinstance(targets, dict) or callable(targets):
            targets = [targets]
        elif targets is None:
            targets = inf_yield_val(targets)

        if isgeneratorfunction(epochs):
            epochs = epochs()

        if (not isinstance(epochs, list) and not isinstance(epochs, tuple)):
            epochs = inf_yield_val(epochs)
        elif epochs is None:
            epochs = inf_yield_val(1)

        skip_initialization = False

        # FIX JDC 12/10/22: PUT with Report HERE, TREATING OUTER LOOP AS RUN, AND RUN AS TRIAL

        for stim_input, stim_target, stim_epoch in zip(inputs, targets, epochs):
            if not callable(stim_input) and 'epochs' in stim_input:
                stim_epoch = stim_input['epochs']

            stim_input, num_input_trials = self._composition._parse_learning_spec(inputs=stim_input,
                                                                                  targets=stim_target,
                                                                                  execution_mode=execution_mode,
                                                                                  context=context)
            if num_trials is None:
                num_trials = num_input_trials

            if minibatch_size == TRAINING_SET:
                minibatch_size = num_trials

            if minibatch_size > num_trials:
                raise Exception("The minibatch size cannot be greater than the number of trials.")

            early_stopper = None
            if patience is not None and not self._is_llvm_mode:
                early_stopper = EarlyStopping(min_delta=min_delta, patience=patience)

            if callable(stim_input) and not isgeneratorfunction(stim_input):
                minibatched_input = self._batch_function_inputs(stim_input,
                                                                stim_epoch,
                                                                num_trials,
                                                                minibatch_size,
                                                                call_before_minibatch=call_before_minibatch,
                                                                call_after_minibatch=call_after_minibatch,
                                                                early_stopper=early_stopper,
                                                                execution_mode=execution_mode,
                                                                context=context)
            else:
                minibatched_input = self._batch_inputs(stim_input,
                                                       stim_epoch,
                                                       num_trials,
                                                       minibatch_size,
                                                       randomize_minibatches,
                                                       call_before_minibatch=call_before_minibatch,
                                                       call_after_minibatch=call_after_minibatch,
                                                       early_stopper=early_stopper,
                                                       execution_mode=execution_mode,
                                                       context=context)

            # The above generators generate:
            # num_trials / batch_size * batch_size * stim_epoch entries
            # unless 'early_stopper' stops the iteration sooner.
            # 'early_stopper' is not allowed in compiled mode.
            # FIXME: Passing the number to Python execution fails several tests.
            # Those test rely on the extra iteration that exits the iterator.
            # (Passing num_trials * stim_epoch + 1 works)
            run_trials = num_trials * stim_epoch if self._is_llvm_mode else None

            self._composition.run(inputs=minibatched_input,
                                  num_trials=run_trials,
                                  skip_initialization=skip_initialization,
                                  skip_analyze_graph=True,
                                  execution_mode=execution_mode,
                                  context=context,
                                  **kwargs)
            skip_initialization = True

        num_epoch_results = num_trials // minibatch_size # number of results expected from final epoch
        # return self._composition.parameters.results.get(context)[-1 * num_epoch_results:]
        # assign results from last *epoch* to learning_results
        self._composition.parameters.learning_results._set(
            self._composition.parameters.results.get(context)[-1 * num_epoch_results:], context)
        # return result of last *trial* (as usual for a call to run)
        return self._composition.parameters.results.get(context)[-1]

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
