# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* AutodiffComposition *************************************************


from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.compositions.composition import CompositionError
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import SOFT_CLAMP
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core import llvm as pnlvm
import copy
import numpy as np

from collections.abc import Iterable
from toposort import toposort

import logging
try:
    import torch
    from torch import nn
    import torch.optim as optim
    from psyneulink.library.compositions.pytorchmodelcreator import PytorchModelCreator
    torch_available = True
except ImportError:
    torch_available = False

logger = logging.getLogger(__name__)


__all__ = [
    "PytorchModelTrainer"
    ]


class PytorchModelTrainer(Composition):

    class Parameters(Composition.Parameters):
        optimizer = None
        learning_rate = .001
        losses = None
        patience = None
        min_delta = 0
        pytorch_representation = None

    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    def __init__(self,
                 autodiffcomposition):

        self.autodiffcomposition = autodiffcomposition
    
    # CLEANUP: move some of what's done in the methods below to a "validate_params" type of method
    def _build_pytorch_representation(self, execution_id = None):
        if self.scheduler is None:  # if learning_enabled has never been run yet
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.execution_sets is None:
            self.execution_sets = list(self.scheduler.run())
        if self.parameters.pytorch_representation._get(execution_id) is None:
            model = PytorchModelCreator(self.graph_processing,
                                        self.param_init_from_pnl,
                                        self.execution_sets,
                                        self.device,
                                        execution_id,
                                        composition = self)
            self.parameters.pytorch_representation._set(model, execution_id)

        # Set up optimizer function
        old_opt = self.parameters.optimizer._get(execution_id)
        if old_opt is not None:
            logger.warning("Overwriting optimizer for AutodiffComposition {}! Old optimizer: {}".format(
                self, old_opt))
        opt = self._make_optimizer(self.optimizer_type, self.learning_rate, self.weight_decay, execution_id)
        self.parameters.optimizer._set(opt, execution_id)

        # Set up loss function
        if self.loss is not None:
            logger.warning("Overwriting loss function for AutodiffComposition {}! Old loss function: {}".format(
                self, self.loss))
        self.loss = self._get_loss(self.loss_spec)

    def _make_optimizer(self, optimizer_type, learning_rate, weight_decay, execution_id):
        if not isinstance(learning_rate, (int, float)):
            raise AutodiffCompositionError("Learning rate must be an integer or float value.")
        if optimizer_type not in ['sgd', 'adam']:
            raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                           "Currently, Stochastic Gradient Descent and Adam are the only available "
                                           "optimizers (specified as 'sgd' or 'adam').")
        params = self.parameters.pytorch_representation._get(execution_id).parameters()
        if optimizer_type == 'sgd':
            return optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    def _get_loss(self, loss_spec):
        if not isinstance(self.loss_spec, str):
            return self.loss_spec
        elif loss_spec == 'mse':
            return nn.MSELoss(reduction='sum')
        elif loss_spec == 'crossentropy':
            return nn.CrossEntropyLoss(reduction='sum')
        elif loss_spec == 'l1':
            return nn.L1Loss(reduction='sum')
        elif loss_spec == 'nll':
            return nn.NLLLoss(reduction='sum')
        elif loss_spec == 'poissonnll':
            return nn.PoissonNLLLoss(reduction='sum')
        elif loss_spec == 'kldiv':
            return nn.KLDivLoss(reduction='sum')
        else:
            raise AutodiffCompositionError("Loss type {} not recognized. Loss argument must be a string or function. "
                                           "Currently, the recognized loss types are Mean Squared Error, Cross Entropy,"
                                           " L1 loss, Negative Log Likelihood loss, Poisson Negative Log Likelihood, "
                                           "and KL Divergence. These are specified as 'mse', 'crossentropy', 'l1', "
                                           "'nll', 'poissonnll', and 'kldiv' respectively.".format(loss_spec))

    def _has_required_keys(self, input_dict):
        required_keys = {"inputs", "targets"}
        return required_keys.issubset(set(input_dict.keys()))

    def _adjust_stimulus_dict(self, inputs):
        if self.learning_enabled:
            if isinstance(inputs, dict):
                if self._has_required_keys(inputs):
                    return [inputs]
                raise AutodiffCompositionError("Invalid input specification.")
            elif isinstance(inputs, list):
                for input_dict in inputs:
                    if not self._has_required_keys(input_dict):
                        raise AutodiffCompositionError("Invalid input specification.")
                return inputs
        return super(AutodiffComposition, self)._adjust_stimulus_dict(inputs)

    # performs forward computation for one input
    def autodiff_processing(self, inputs, execution_id=None, do_logging=False, scheduler=None,bin_execute=False):
        pytorch_representation = self.parameters.pytorch_representation._get(execution_id)
        # run the model on inputs - switch autograd off for this (we don't need it)
        with torch.no_grad():
            tensor_outputs = pytorch_representation.forward(inputs, execution_id=execution_id, do_logging=do_logging, scheduler=scheduler)

        # get outputs back into numpy
        outputs = []
        for i in range(len(tensor_outputs)):
            outputs.append(tensor_outputs[i].numpy().copy())

        return outputs

    # performs learning/training on all input-target pairs it recieves for given number of epochs
    def autodiff_training(self, inputs, targets, epochs, execution_id=None, do_logging=False, scheduler=None,bin_execute=False):

        # FIX CW 11/1/18: this value of num_inputs assumes all inputs have same length, and that the length of
        # the input for an origin component equals the number of desired trials. We could clean this up
        # by perhaps using modular arithmetic on t, or by being more explicit about number of desired trials
        first_input_value = list(inputs.values())[0]
        num_inputs = len(first_input_value)

        patience = self.parameters.patience._get(execution_id)

        if patience is not None:
            # set up object for early stopping
            early_stopper = EarlyStopping(patience=patience, min_delta=self.parameters.min_delta._get(execution_id))

        # if training over trial sets in random order, set up array for mapping random order back to original order
        if self.randomize:
            rand_train_order_reverse = np.zeros(num_inputs)

        # get total number of output neurons from the dimensionality of targets on the first trial
        # (this is for computing average loss across neurons on each trial later)
        out_size = 0
        for target in targets.values():
            out_size += len(target)

        # iterate over epochs
        for epoch in range(epochs):

            # if training in random order, generate random order and set up mapping
            # from random order back to original order
            if self.randomize:
                rand_train_order = np.random.permutation(num_inputs)
                rand_train_order_reverse[rand_train_order] = np.arange(num_inputs)

            # set up array to keep track of losses on epoch
            curr_losses = np.zeros(num_inputs)

            # reset temporary list to keep track of most recent outputs
            outputs = []

            self.parameters.pytorch_representation._get(execution_id).detach_all()
            # self.parameters.pytorch_representation._get(execution_id).reset_all()

            # iterate over inputs, targets
            for t in range(num_inputs):

                if self.randomize:
                    input_index = rand_train_order[t]
                else:
                    input_index = t
                curr_tensor_inputs = {}
                curr_tensor_targets = {}
                for component in inputs.keys():
                    input = inputs[component][input_index]
                    curr_tensor_inputs[component] = torch.tensor(input, device=self.device).double()
                for component in targets.keys():
                    target = targets[component][input_index]
                    curr_tensor_targets[component] = torch.tensor(target, device=self.device).double()

                # do forward computation on current inputs
                curr_tensor_outputs = self.parameters.pytorch_representation._get(execution_id).forward(
                    curr_tensor_inputs,
                    execution_id,
                    do_logging,
                    scheduler=scheduler,
                )

                # compute total loss across output neurons for current trial
                curr_loss = torch.zeros(1).double()
                for component in curr_tensor_outputs.keys():
                    # possibly add custom loss option, which is a loss function that takes many args
                    # (outputs, targets, weights, and more) and returns a scalar
                    curr_loss += self.loss(curr_tensor_outputs[component], curr_tensor_targets[component])

                # save average loss across all output neurons on current trial
                curr_losses[t] = (curr_loss[0].item())/out_size

                optimizer = self.parameters.optimizer._get(execution_id)

                # backpropagate to compute gradients and perform learning update for parameters
                optimizer.zero_grad()
                curr_loss = curr_loss/2
                if self.force_no_retain_graph:
                    curr_loss.backward(retain_graph=False)
                else:
                    curr_loss.backward(retain_graph=True)
                self.parameters.pytorch_representation._get(execution_id).copy_weights_to_psyneulink(execution_id)
                optimizer.step()

                # save outputs of model if this is final epoch
                curr_output_list = []
                for input_state in self.output_CIM.input_states:
                    assert(len(input_state.all_afferents) == 1)  # CW 12/05/18, this assert may eventually be outdated
                    component = input_state.all_afferents[0].sender.owner
                    curr_output_list.append(curr_tensor_outputs[component].detach().numpy().copy())
                # for component in curr_tensor_outputs.keys():
                #     curr_output_list.append(curr_tensor_outputs[component].detach().numpy().copy())
                outputs.append(curr_output_list)

                scheduler.get_clock(execution_id)._increment_time(TimeScale.TRIAL)

            # save average loss on the current epoch
            average_loss = np.mean(curr_losses)
            self.parameters.losses._get(execution_id).append(average_loss)

            # update early stopper with most recent average loss
            if self.parameters.patience._get(execution_id) is not None:
                should_stop = early_stopper.step(average_loss)
                if should_stop:
                    logger.warning('Stopped training early after {} epochs'.format(epoch))
                    if self.randomize:
                        outputs_list = [None] * len(outputs)
                        for i in range(len(outputs)):
                            outputs_list[i] = outputs[int(rand_train_order_reverse[i])]
                        return outputs_list
                    else:
                        return outputs

        if self.randomize:  # save outputs in a list in correct order, return them
            outputs_list = [None] * len(outputs)
            for i in range(len(outputs)):
                outputs_list[i] = outputs[int(rand_train_order_reverse[i])]
            return outputs_list
        else:
            return outputs

    @property
    def _bin_exec_func(self):
        if self.__bin_exec_func is None:
            with pnlvm.LLVMBuilderContext.get_global() as ctx:
                self.__bin_exec_func = ctx.gen_autodiffcomp_exec(self)
        return self.__bin_exec_func
    
    def _gen_llvm_function(self):
        return self._bin_exec_func

    def execute(self,
                inputs=None,
                autodiff_stimuli=None,
                do_logging=False,
                scheduler_processing=None,
                termination_processing=None,
                call_before_time_step=None,
                call_before_pass=None,
                call_after_time_step=None,
                call_after_pass=None,
                execution_id=None,
                base_execution_id=None,
                clamp_input=SOFT_CLAMP,
                targets=None,
                runtime_params=None,
                skip_initialization=False,
                bin_execute=False,
                context=None
                ):
        execution_id = self._assign_execution_ids(execution_id)
        self._assign_context_values(execution_id=execution_id, composition=self, propagate=True)

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        scheduler_processing._init_clock(execution_id, base_execution_id)

        if self.learning_enabled:
            # TBI: How are we supposed to use base_execution_id and statefulness here?
            # TBI: can we call _build_pytorch_representation in _analyze_graph so that pytorch
            # model may be modified between runs?

            self._analyze_graph()  # ADDED by CW 12/17/18: unsure if correct here

            self._build_pytorch_representation(execution_id)

            autodiff_inputs = inputs["inputs"]
            autodiff_targets = inputs["targets"]
            autodiff_epochs = 1
            if "epochs" in inputs:
                autodiff_epochs = inputs["epochs"]

            output = self.autodiff_training(autodiff_inputs, autodiff_targets, autodiff_epochs, execution_id, do_logging, scheduler_processing)
            ctx = self.output_CIM.parameters.context._get(execution_id)
            # new_ctx = copy.deepcopy(ctx)
            # new_ctx.execution_phase = ContextFlags.PROCESSING
            # self.output_CIM.parameters.context._set(new_ctx, execution_id=execution_id)
            if ctx is not None:  # HACK: CW 12/18/18 for some reason context isn't set correctly
                ctx.execution_phase = ContextFlags.PROCESSING
            # note that output[-1] might not be the truly most recent value
            # HACK CW 2/5/19: the line below is a hack. In general, the output_CIM of an AutodiffComposition
            # is not having its parameters populated correctly, and this should be fixed in the long run.
            self.output_CIM.execute(input=output[-1], execution_id=execution_id, context=ContextFlags.PROCESSING)

            return output

        return super(AutodiffComposition, self).execute(inputs=inputs,
                                                        scheduler_processing=scheduler_processing,
                                                        termination_processing=termination_processing,
                                                        call_before_time_step=call_before_time_step,
                                                        call_before_pass=call_before_pass,
                                                        call_after_time_step=call_after_time_step,
                                                        call_after_pass=call_after_pass,
                                                        execution_id=execution_id,
                                                        base_execution_id=base_execution_id,
                                                        clamp_input=clamp_input,
                                                        runtime_params=runtime_params,
                                                        skip_initialization=skip_initialization,
                                                        bin_execute=bin_execute,
                                                        context=context)

    # what the user calls for doing processing/training, similar to the run function of the normal composition
    def run(
        self,
        inputs=None,
        do_logging=False,
        scheduler_processing=None,
        termination_processing=None,
        execution_id=None,
        num_trials=1,
        call_before_time_step=None,
        call_after_time_step=None,
        call_before_pass=None,
        call_after_pass=None,
        call_before_trial=None,
        call_after_trial=None,
        clamp_input=SOFT_CLAMP,
        bin_execute=False,
        initial_values=None,
        reinitialize_values=None,
        runtime_params=None,
        context=None):
        # TBI: Handle trials, timesteps, etc
        execution_id = self._assign_execution_ids(execution_id)
        self._assign_context_values(execution_id=execution_id, composition=self, propagate=True)

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        scheduler_processing._init_clock(execution_id)

        if self.learning_enabled:

            self._analyze_graph()

            if self.refresh_losses or (self.parameters.losses._get(execution_id) is None):
                self.parameters.losses._set([], execution_id)
            adjusted_stimuli = self._adjust_stimulus_dict(inputs)
            if num_trials is None:
                num_trials = len(adjusted_stimuli)

            results = []
            for trial_num in range(num_trials):
                stimulus_index = trial_num % len(adjusted_stimuli)
                trial_output = self.execute(
                    inputs=adjusted_stimuli[stimulus_index],
                    execution_id=execution_id,
                    do_logging=do_logging,
                    bin_execute = bin_execute
                )
                results.append(trial_output)

        else:
            results = super(AutodiffComposition, self).run(inputs=inputs,
                                                    scheduler_processing=scheduler_processing,
                                                    termination_processing=termination_processing,
                                                    execution_id=execution_id,
                                                    num_trials=num_trials,
                                                    call_before_time_step=call_before_time_step,
                                                    call_after_time_step=call_after_time_step,
                                                    call_before_pass=call_before_pass,
                                                    call_after_pass=call_after_pass,
                                                    call_before_trial=call_before_trial,
                                                    call_after_trial=call_after_trial,
                                                    clamp_input=clamp_input,
                                                    bin_execute=bin_execute,
                                                    initial_values=initial_values,
                                                    reinitialize_values=reinitialize_values,
                                                    runtime_params=runtime_params,
                                                    context=context)

        scheduler_processing.get_clock(execution_id)._increment_time(TimeScale.RUN)
        return results


    # gives user weights and biases of the model (from the pytorch representation)
    def get_parameters(self, execution_id=NotImplemented):
        if execution_id is NotImplemented:
            execution_id = self.default_execution_id

        pytorch_representation = self.parameters.pytorch_representation._get(execution_id)

        if pytorch_representation is None:
            raise AutodiffCompositionError("{0} has not been run yet so parameters have not been created "
                                           "in Pytorch."
                                           .format(self.name))

        weights = pytorch_representation.get_weights_for_projections()
        biases = pytorch_representation.get_biases_for_mechanisms()

        return weights, biases

    def _get_param_struct_type(self, ctx):
        return self.autodiffcomposition._get_param_struct_type(ctx)

    def _get_param_initializer(self, execution_id, simulation=False):
        return self.autodiffcomposition._get_param_initializer(execution_id,simulation)
