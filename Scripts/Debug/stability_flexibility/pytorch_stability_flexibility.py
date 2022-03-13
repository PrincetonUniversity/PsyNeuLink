from typing import Union, Iterable, Tuple, Optional, Callable

import numpy as np
import torch

from pytorch_lca import LCALayer

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


class DriftDiffusionModel(torch.nn.Module):
    def forward(self,
                activities: torch.Tensor,
                rate: float = 1.0,
                threshold: float = 1.0,
                noise: float = 1.0,
                time_step_size: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        A model that simulates many instances of a simple noisy drift diffusion model in parallel.

        Args:
            activities: The current actitivies of the DDM
            rate: The drift rate for each particle.
            threshold: The threshold that a particle must reach to stop integration.
            noise: The standard deviation of the Gaussian noise added to each particles position at each time step.
            time_step_size: The time step size (in seconds) for the integration process.

        Returns:
            A two element tuple containing the reaction times and the decisions
        """

        if threshold is not None:
            active = torch.abs(activities) < threshold
        else:
            active = torch.ones(size=activities.size(), dev=dev)

        dw = torch.normal(mean=rate * time_step_size * active, std=noise * active)
        activities = activities + dw * np.sqrt(time_step_size)

        return activities, active


# Model Parameters
GAIN = 1.0
LEAK = 1.0
COMP = 7.5
AUTOMATICITY = 0.15  # Automaticity Weight
STARTING_VALUE = 0.0  # Starting Point
THRESHOLD = 0.5  # Threshold
LCA_NOISE = None  # Noise
DDM_NOISE = 0.1
SCALE = 1.0  # Scales DDM inputs so threshold can be set to 1
NON_DECISION_TIME = 0.1

# Number of simulations to run. Each simulation is independent.
NUM_SIMULATIONS = 1000

# Run the model
N_TIME_STEPS = 3000
TIME_STEP_SIZE = 0.01

stimuli = np.array([[-1, -1],
                    [-1, -1],
                    [1, 1],
                    [-1, 1],
                    [-1, -1],
                    [-1, 1]])

tasks = np.array([[0, 1],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [0, 1]])

lca = LCALayer()
ddm = DriftDiffusionModel()

cue_stimulus_intervals = np.array([[0.3], [0.3], [0.3], [0.3], [0.3], [0.3]])

NUM_TRIALS = len(stimuli)

# Initialize the LCA task activities, these are maintained throughout the whole
# experiment.
lca_activities = torch.zeros(size=(NUM_SIMULATIONS, 2), device=dev)

for trial_idx in range(NUM_TRIALS):

    # Reset DDM activities for this trial.
    ddm_activities = torch.ones(size=(NUM_SIMULATIONS,), device=dev) * STARTING_VALUE
    rts = torch.zeros(size=(NUM_SIMULATIONS,), device=dev)

    stimulus = torch.from_numpy(stimuli[trial_idx]).float().to(dev)
    task = torch.from_numpy(tasks[trial_idx]).float().to(dev)
    csi = torch.from_numpy(cue_stimulus_intervals[trial_idx]).float().to(dev)

    # Compute the Automaticity-weighted Stimulus Input
    auto_weight_stim = torch.sum(stimulus * AUTOMATICITY)

    # Simulate N time steps of the model for this trial
    for time_i in range(N_TIME_STEPS):

        # Compute the LCA task activities
        lca_activities, _ = lca(input=task,
                                activities=lca_activities,
                                threshold=None,
                                leak=LEAK,
                                competition=COMP,
                                self_excitation=0.0,
                                noise=None,
                                time_step_size=TIME_STEP_SIZE)

        # If the Cue Stimulus Interval time has passed, start the decision process.
        if time_i * TIME_STEP_SIZE > csi:
            # Compute the drift rate for the DDM from the task activations and the stimulus
            non_automatic_component = torch.sum(lca_activities * stimulus, dim=1)
            drift_rate = non_automatic_component + auto_weight_stim

            ddm_activities, ddm_active = ddm(activities=ddm_activities, rate=drift_rate, threshold=THRESHOLD,
                                             noise=DDM_NOISE, time_step_size=TIME_STEP_SIZE)

            # Compute the reaction time for each simulation.
            rts = rts + ddm_active

    # Compute reaction times in seconds for these trials
    rts = (NON_DECISION_TIME + rts * TIME_STEP_SIZE)

    decisions = torch.ones(size=(NUM_SIMULATIONS,), device=dev)
    decisions[ddm_activities <= -THRESHOLD] = 0
