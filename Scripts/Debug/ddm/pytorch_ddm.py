#%%
import time
import torch
from torch import nn

from typing import Tuple

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"


class DriftDiffusionModel(torch.nn.Module):
    def forward(self,
                starting_value: float = 0.0,
                rate: float = 1.0,
                non_decision_time: float = 0.0,
                threshold: float = 1.0,
                noise: float = 1.0,
                time_step_size: float = 0.01,
                num_walkers: int = 1000,
                dev: str = "cuda:0") -> Tuple[torch.Tensor, torch.Tensor]:

        """
        A model that simulates many instances of a simple noisy drift diffusion model in parallel.

        Args:
            starting_value: The starting value of each particle in the model.
            rate: The drift rate for each particle.
            non_decision_time: A constant amount of time added to each reaction time that signifies automatic processing
                times of stimuli.
            threshold: The threshold that a particle must reach to stop integration.
            noise: The standard deviation of the Gaussian noise added to each particles position at each time step.
            time_step_size: The time step size (in seconds) for the integration process.
            num_walkers: The number of particles to simulate.
            dev: The device the model should be run on.

        Returns:
            A two element tuple containing the reaction times and the decisions
        """

        particle = torch.ones(size=(num_walkers,), device=dev) * starting_value
        active = torch.ones(size=(num_walkers,), dtype=torch.bool, device=dev)
        rts = torch.zeros(size=(num_walkers,), device=dev)

        for i in range(3000):
            #dw = torch.distributions.Normal(loc=rate * time_step_size * active, scale=noise * active).rsample()
            dw = torch.normal(mean=rate * time_step_size * active, std=noise * active)
            particle = particle + dw * torch.sqrt(time_step_size)
            rts = rts + active
            active = torch.abs(particle) < threshold

        rts = (non_decision_time + rts * time_step_size)

        decisions = torch.ones(size=(num_walkers,), device=dev)
        decisions[particle <= -threshold] = 0

        return rts, decisions


ddm_params = dict(starting_value=0.0, rate=0.3, non_decision_time=0.15, threshold=0.6, noise=1.0, time_step_size=0.001)

NUM_WALKERS = 1000000

# Move params to device
for key, val in ddm_params.items():
    ddm_params[key] = torch.tensor(val).to(dev)

#%%

t0 = time.time()
ddm_model = DriftDiffusionModel()
rts, decision = ddm_model(**ddm_params, num_walkers=NUM_WALKERS, dev=dev)
rts = rts.to("cpu")
decision = decision.to("cpu")
print(f"PyTorch Elapsed: {1000.0 * (time.time() - t0)} milliseconds")

#%%

# JIT
jit_ddm_model = torch.jit.script(ddm_model)

#%%
# rts, decision = jit_ddm_model(**ddm_params, num_walkers=NUM_WALKERS)
#
# NUM_TIMES = 50
# t0 = time.time()
# for i in range(NUM_TIMES):
#     rts, decision = jit_ddm_model(**ddm_params, num_walkers=NUM_WALKERS)
#     rts = rts.to("cpu")
#     decision = decision.to("cpu")
# print(f"JIT Elapsed: {1000 * ((time.time() - t0) / NUM_TIMES)} milliseconds")

#%%

jit_ddm_model.save("ddm.pt")

#%%
# with open('ddm.onnx', 'wb') as file:
#     torch.onnx.export(model=torch.jit.script(DriftDiffusionModel()),
#                       args=tuple(ddm_params.values()) + (torch.tensor(NUM_WALKERS),),
#                       example_outputs=(rts, decision),
#                       f=file,
#                       verbose=True,
#                       opset_version=12)

#%%
# import seaborn as sns
# sns.kdeplot(rts)
