import torch
import onnx
import onnxruntime as ort

from typing import Union, Iterable, Tuple, Optional, Callable, Any


if torch.cuda.is_available():
    def_dev = torch.device("cuda:0")
else:
    def_dev = torch.device("cpu")


def scalar(val: torch.Tensor):
    """Helper function to get scalars as Python numbers"""
    return val.item()


class LCALayer(torch.nn.Module):
    def __init__(
        self,
        threshold: Union[float, torch.FloatTensor, None] = torch.tensor(1.0),
        leak: Union[torch.FloatTensor, float] = torch.tensor(0.1),
        competition: Union[torch.FloatTensor, float] = torch.tensor(0.1),
        self_excitation: Union[torch.FloatTensor, float] = torch.tensor(0.0),
        non_decision_time: Union[torch.FloatTensor, float] = torch.tensor(0.0),
        activation_function: Callable = torch.relu,
        noise: Union[float, torch.Tensor, None] = torch.tensor(1.0),
        time_step_size: Union[torch.FloatTensor, float] = torch.tensor(0.01),
    ):
        """
        An implementation of a Leaky Competing Accumulator as a layer. Each call to forward of this module only
        implements one time step of the integration. See module LCAModel if you want to simulate an LCA to completion.

        Args:
            threshold: The threshold that accumulators must reach to stop integration. If None, accumulators will
                never stop integrating even when the pass the threshold.
            leak:  The decay rate, which reflects leakage of the activation.
            competition: The weight to apply for inhibitory influence from other activations. Positive values lead
                to inhibitory effects from other accumulators.
            self_excitation: The weight to apply for the scaling factor for the recurrent excitation
            non_decision_time: The time that should be added to reaction times to account for stimulus encoding and
                response generation. This parameter shifts the results reactions times by and offset. This parameter
                is not actually used by LCALayer since the forward method only computes a timestep at a time. However,
                this is a common LCA model parameter so it should be stored with the others. It is added to final
                reaction time generated from LCAModel.forward.
            activation_function: The non-linear function to apply to pre activities to get activities. This is
                torch.relu by default, but can be any callable.
            noise: The standard deviation of the Gaussian noise added to each particles position at each time step.
            time_step_size: The time step size (in seconds) for the integration process.

        """
        super().__init__()

        require_grad = False

        self.leak = leak
        self.competition = competition
        self.self_excitation = self_excitation
        self.noise = noise
        self.time_step_size = time_step_size
        self.non_decision_time = non_decision_time
        self._sqrt_step_size = torch.sqrt(
            torch.tensor(0.001, requires_grad=require_grad).to(leak.device)
        )
        self.threshold = threshold
        self.activation_function = activation_function

    def forward(
        self,
        ff_input: torch.Tensor,
        pre_activities: torch.Tensor,
        activities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute one time step of integration for a batch of independent leaky competing accumulator
        instances.

        Args:
            ff_input: The current input activity for the accumulators
            pre_activities: The activity of the accumulator without the application of the non-linearity
                activation_func. A 2D tensors of shape [batch_size, num_accumulators]. batch_size in
                this case is each independent accumulator model instance. Or in other words, a simulation of the model.
            activities: The current activity of each accumulator instance. This is the value after
                the non-linearity activation_func has been applied to pre_activities. A 2D tensors of shape
                [batch_size, num_accumulators]. batch_size in this case is each independent accumulator
                model instance. Or in other words, a simulation of the model.

        Returns:
            The current output activities of the accumulators, of same shape as activities.
        """
        num_simulations, num_lca_dim = activities.shape

        dev = self.leak.device

        # Mark all accumulators as active by default
        active = torch.ones(size=(num_simulations, 1), device=dev)

        # If threshold is provided, only integrate accumulators until they reach the threshold.
        # if self.threshold is not None:
        #     active = torch.all(
        #         torch.abs(activities) < self.threshold, dim=1, keepdim=True
        #     )

        # Construct a gamma matrix, this is multiplied by each accumulator vector to compute
        # the competitive inhibition and self excitation between units.
        gamma = (torch.eye(num_lca_dim, device=dev) == 0.0) * self.competition
        gamma.fill_diagonal_(-self.self_excitation)

        # Perform one time step of the integration
        pre_activities = (
            pre_activities
            + (ff_input - self.leak * pre_activities - torch.matmul(activities, gamma))
            * active
            * self.time_step_size
        )

        # If standard deviation of noise is provided. Generate a noise for each accumulator.
        # Only active accumulators will get noise
        if self.noise is not None:
            dw = torch.normal(
                mean=torch.zeros(activities.size(), device=dev),
                std=active * self.noise * torch.ones(activities.size(), device=dev),
            )
            pre_activities = pre_activities + dw * self._sqrt_step_size

        # Calculate the post activation function activities. Don't overwrite pre_activities, we will need these for
        # the next timestep.
        activities = self.activation_function(pre_activities)

        return pre_activities, activities, active


class LCAModel(torch.nn.Module):
    def __init__(
        self,
        lca_layer: LCALayer,
        num_lca_dim: int,
        num_simulations: int = 10000,
        num_time_steps: int = 3000,
    ):
        """
        A model that simulates a leaky competing accumulator model (Usher and McClelland).

        References:

        Usher M, McClelland JL. The time course of perceptual choice: the leaky, competing accumulator model.
        Psychol Rev. 2001 Jul;108(3):550-92. doi: 10.1037/0033-295x.108.3.550. PMID: 11488378.

        Args:
            lca_layer: The LCALayer that computes the integration for each timestep.
            num_lca_dim: The number of LCA units or accumulators.
            num_simulations: The number of parallel independent simulations to run.
            num_time_steps: The total number of time steps to run.

        """

        super().__init__()

        self.lca_layer = lca_layer
        self.num_lca_dim = num_lca_dim
        self.num_simulations = num_simulations
        self.num_time_steps = num_time_steps

    def forward(self, ff_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            ff_input: A vector specifying the feed forward input for each accumulator.
                This is typically a representation of the stimulus, task, or both.

        Returns:
            A tuple with two vectors, the activations before and after applying the non-linear
            activation function.
        """
        dev = ff_input.device

        # Mark all accumulators as active by default
        active = torch.ones(size=[int(self.num_simulations), 1], device=dev)

        pre_activities = torch.zeros(
            size=[int(self.num_simulations), int(self.num_lca_dim)], device=dev
        )
        activities = torch.zeros(
            size=[int(self.num_simulations), int(self.num_lca_dim)], device=dev
        )

        rts = torch.zeros(size=[int(self.num_simulations), 1], device=dev)

        # Simulate N time steps of the model. This could be done with a while(active) type loop but this is slightly
        # faster actually on the GPU.
        for time_i in range(int(self.num_time_steps)):

            # Compute the LCA activities
            pre_activities, activities, active = self.lca_layer(
                ff_input=ff_input, pre_activities=pre_activities, activities=activities
            )

            # Only increment reaction time for active simulations
            rts = rts + active

        # Convert the time step index to actual time
        rts = rts * self.lca_layer.time_step_size

        # Include the non-decision time in the reaction time.
        rts = rts + self.lca_layer.non_decision_time

        # Figure out which accumulator crossed the threshold, this is the decision
        torch.max(activities, dim=1)

        max_values, decisions = torch.max(activities, dim=1, keepdim=True)

        # Find any simulations that had multiple accumulators cross the threshold at the same time.
        # Exclude these for simplicity.
        # good = torch.logical_and(
        #     torch.sum(activities == max_values, dim=1) == 1, ~torch.squeeze(active)
        # )
        # decisions = decisions[good]
        # rts = rts[good]

        return rts, decisions

if __name__ == "__main__":

    lca_params = dict(
        threshold=0.06,
        leak=10.0,
        competition=0.0,
        self_excitation=6.2,
        non_decision_time=0.0,
        noise=0.1,
        # noise=None,
        time_step_size=0.001,
    )

    # Convert all the lca parameters to torch.FloatTensors
    lca_params = {k: torch.tensor(v, device=def_dev) for k, v in lca_params.items()}


    lca = LCAModel(
        lca_layer=LCALayer(
            **lca_params,
            activation_function=torch.relu,
        ),
        num_lca_dim=torch.tensor(2),
        num_simulations=torch.tensor(50000),
        num_time_steps=torch.tensor(3000),
    )

    # Compile things for a bit of a performance boost
    lca = torch.jit.script(lca)

    # Create some input for the LCA, typically this would be generated from another part of the model
    # that processes stimulus or tasks information into an N sized vector, where N is the number of
    # dimensions in the LCA accumulator.
    input_array = [1.02, 1.02 + 0.02]
    ff_input = torch.Tensor(input_array).to(def_dev)

    # Run things to get some example outputs.
    outputs = lca(ff_input)

    torch.onnx.export(lca,  # model being run
                      ff_input,  # model input (or a tuple for multiple inputs)
                      "lca.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['ff_input'],  # the model's input names
                      output_names=['rts', 'decisions'],  # the model's output names
                      #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                      example_outputs=outputs,
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #              'output': {0: 'batch_size'}}
                      )

    onnx_model = onnx.load("lca.onnx")
    onnx.checker.check_model(onnx_model)
