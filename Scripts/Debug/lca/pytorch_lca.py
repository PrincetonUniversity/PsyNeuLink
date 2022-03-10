import torch
import numpy as np
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from typing import Union, Iterable, Tuple, Optional, Callable

import psyneulink as pnl


if torch.cuda.is_available():
    def_dev = "cuda:0"
else:
    def_dev = "cpu"


class LCALayer(torch.nn.Module):
    def __init__(
        self,
        threshold: Union[float, None] = 1.0,
        leak: float = 0.1,
        competition: float = 0.1,
        self_excitation: float = 0.0,
        non_decision_time: float = 0.0,
        activation_function: Callable = torch.relu,
        noise: Union[float, torch.Tensor, None] = 1.0,
        time_step_size: float = 0.01,
        dev: Optional[Union[str, torch.device]] = "cpu"
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
            dev: Device to run compute on.

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
            torch.tensor(0.001, requires_grad=require_grad).to(dev)
        )
        self.threshold = threshold
        self.activation_function = activation_function
        self.dev = dev

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

        # Mark all accumulators as active by default
        active = torch.ones(size=(num_simulations, 1), device=self.dev)

        # If threshold is provided, only integrate accumulators until they reach the threshold.
        if self.threshold is not None:
            active = torch.all(
                torch.abs(activities) < self.threshold, dim=1, keepdim=True
            )

        # Construct a gamma matrix, this is multiplied by each accumulator vector to compute
        # the competitive inhibition and self excitation between units.
        gamma = (torch.eye(num_lca_dim, device=self.dev) == 0.0) * self.competition
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
                mean=torch.zeros(activities.size(), device=self.dev),
                std=active * self.noise * torch.ones(activities.size(), device=self.dev),
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
        save_activities: bool = False,
        dev: Optional[Union[str, torch.device]] = "cpu"
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
            save_activities: Should activities be saved. If True, make sure the number of simulations is low.

        """

        super().__init__()

        self.lca_layer = lca_layer
        self.num_lca_dim = num_lca_dim
        self.num_simulations = num_simulations
        self.num_time_steps = num_time_steps
        self.save_activities = save_activities
        self.dev = dev

    def forward(self, ff_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            ff_input: A vector specifying the feed forward input for each accumulator.
                This is typically a representation of the stimulus, task, or both.

        Returns:
            A tuple with two vectors, the activations before and after applying the non-linear
            activation function.
        """

        # Mark all accumulators as active by default
        active = torch.ones(size=(self.num_simulations, 1), device=self.dev)

        if self.save_activities:
            pre_activities = torch.zeros(
                size=(self.num_simulations, self.num_lca_dim, self.num_time_steps + 1),
                device=self.dev,
            )
            activities = torch.zeros(
                size=(self.num_simulations, self.num_lca_dim, self.num_time_steps + 1),
                device=self.dev,
            )
        else:
            pre_activities = torch.zeros(
                size=(self.num_simulations, self.num_lca_dim), device=self.dev
            )
            activities = torch.zeros(
                size=(self.num_simulations, self.num_lca_dim), device=self.dev
            )

        rts = torch.zeros(size=(self.num_simulations, 1), device=self.dev)

        # Simulate N time steps of the model. This could be done with a while(active) type loop but this is slightly
        # faster actually on the GPU.
        for time_i in range(self.num_time_steps):

            # Compute the LCA activities
            if self.save_activities:
                (
                    pre_activities[:, :, time_i + 1],
                    activities[:, :, time_i + 1],
                    active,
                ) = self.lca_layer(
                    ff_input=ff_input,
                    pre_activities=pre_activities[:, :, time_i],
                    activities=activities[:, :, time_i],
                )
            else:
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

        if not self.save_activities:
            max_values, decisions = torch.max(activities, dim=1, keepdim=True)

            # Find any simulations that had multiple accumulators cross the threshold at the same time.
            # Exclude these for simplicity.
            good = torch.logical_and(
                torch.sum(activities == max_values, dim=1) == 1, ~torch.squeeze(active)
            )
            decisions = decisions[good]
            rts = rts[good]

        else:
            decisions = torch.argmax(activities[:, :, self.num_time_steps], dim=1)

        if self.save_activities:
            return pre_activities, activities
        else:
            return rts, decisions


def make_pnl_lca(
    num_lca_dim: int = 2,
    threshold: Union[float, None] = 1.0,
    leak: float = 0.1,
    competition: float = 0.1,
    self_excitation: float = 0.0,
    non_decision_time: float = 0.0,
    activation_function: Callable = pnl.ReLU,
    noise: Union[float, torch.Tensor, None] = 1.0,
    time_step_size: float = 0.01,
):

    if noise is not None:
        noise_func = lambda: rng.normal(loc=0.0, scale=noise)
    else:
        noise_func = 0.0

    lca = pnl.LCAMechanism(
        default_variable=[[0.0 for _ in range(num_lca_dim)]],
        size=num_lca_dim,
        threshold=threshold,
        function=activation_function,
        leak=leak,
        competition=competition,
        self_excitation=self_excitation,
        noise=noise_func,
        time_step_size=time_step_size,
        termination_measure=max,
        termination_comparison_op=">=",
        reset_stateful_function_when=pnl.AtTrialStart(),
        # execute_until_finished=True,
        # max_executions_before_finished=sys.maxsize,
        name="LCA",
    )

    lca.set_log_conditions([pnl.RESULT])

    comp = pnl.Composition(name="LCA-Comp")
    comp.add_node(lca)

    return comp, lca


if __name__ == "__main__":

    # Set to true if you want to plot an example histories of activations
    save_activities = False

    if save_activities:
        num_simulations = 10
    else:
        num_simulations = 50000

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

    lca = LCAModel(
        lca_layer=LCALayer(
            **lca_params,
            activation_function=torch.relu,
            dev=def_dev
        ),
        num_lca_dim=2,
        num_simulations=num_simulations,
        num_time_steps=3000,
        save_activities=save_activities,
        dev=def_dev
    )

    # Compile things for a bit of a performance boost
    lca = torch.jit.script(lca)

    rng = np.random.RandomState(seed=42)
    comp, pnl_lca_mech = make_pnl_lca(
        num_lca_dim=2, activation_function=pnl.ReLU, **lca_params
    )

    # Create some input for the LCA, typically this would be generated from another part of the model
    # that processes stimulus or tasks information into an N sized vector, where N is the number of
    # dimensions in the LCA accumulator.
    input_array = [1.02, 1.02 + 0.02]
    input_pnl = {pnl_lca_mech: [input_array]}
    ff_input = torch.Tensor(input_array).to(def_dev)

    # Run the PNL LCA
    comp.run(input_pnl, execution_mode=pnl.ExecutionMode.Python, num_trials=10)
    pnl_results = comp.results

    # Get the LCA activities from the log if running in non-compiled mode. Just get one trial's worth
    pnl_activities = np.array(
        [
            t.value.tolist()
            for t in pnl_lca_mech.log.get_logged_entries()["RESULT"]["LCA-Comp"]
            if t.time.trial == 0
        ]
    )

    # Warm things up before running things for timing
    lca(ff_input)
    lca(ff_input)

    t0 = time.time()

    # Run things ten times so we can get a rough average execution time.
    for i in range(10):
        # Run many independent simulations of the same LCA model
        if save_activities:
            pre_activities, activities = lca(ff_input=ff_input)
            activities = activities.to("cpu").numpy()
            pre_activities = pre_activities.to("cpu").numpy()
        else:
            rts, decisions = lca(ff_input=ff_input)
            rts = rts.to("cpu").numpy().flatten()
            decisions = decisions.to("cpu").numpy().flatten()

    t1 = time.time()
    print(f"Average Elapsed: {(t1-t0) / 10.0} seconds")


    if save_activities:
        pre_activities = np.transpose(pre_activities[0, :, :])
        activities = np.transpose(activities[0, :, :])
        rt_i = max(np.argmax(activities, axis=0)) + 1
        activities = activities[0:rt_i, :]
        pre_activities = pre_activities[0:rt_i, :]
        preact_df = pd.DataFrame(
            pre_activities, columns=[f"{i}" for i in range(pre_activities.shape[1])]
        )
        preact_df["name"] = "pre activation"
        preact_df["t"] = np.arange(rt_i) * lca.lca_layer.time_step_size
        act_df = pd.DataFrame(
            activities, columns=[f"{i}" for i in range(activities.shape[1])]
        )
        act_df["name"] = "post activation"
        act_df["t"] = np.arange(rt_i) * lca.lca_layer.time_step_size
        results = pd.concat([preact_df, act_df])
        results = pd.melt(results, id_vars=["name", "t"], var_name="lca_unit")

        # Add the PNL results
        for i in range(lca.num_lca_dim):
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            "name": "PNL post activation",
                            "t": (np.arange(pnl_activities.shape[0]) + 1)
                            * lca.lca_layer.time_step_size,
                            "lca_unit": str(i),
                            "value": pnl_activities[:, i],
                        }
                    ),
                ]
            )

        #%%
        g = sns.FacetGrid(
            results[results["name"].str.contains("post activation")],
            col="lca_unit",
            hue="name",
        )
        g.map_dataframe(sns.lineplot, x="t", y="value")
        g.set_axis_labels("Time (s)", "Activity")
        g.add_legend()

        ax1, ax2 = g.axes[0]

        ax1.axhline(lca.lca_layer.threshold, ls="--")
        ax2.axhline(lca.lca_layer.threshold, ls="--")

        ax1.axvline((rt_i - 1) * lca.lca_layer.time_step_size, ls="--")
        ax2.axvline((rt_i - 1) * lca.lca_layer.time_step_size, ls="--")

        plt.show()
    #%%

    else:
        results = pd.DataFrame({"reaction_time": rts, "decision": decisions})
        sns.kdeplot(data=results, x="reaction_time", hue="decision")
        plt.xlim([0.0, 3.5])
        plt.show()
