#%%
import time
import torch
from torch import nn

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"


def simulate_ddms(starting_value: float = 0.0,
                  rate: float = 1.0,
                  non_decision_time: float = 0.0,
                  threshold: float = 1.0,
                  noise: float = 1.0,
                  time_step_size: float = 0.01,
                  num_walkers: int = 1000,
                  dev: str = "cuda:0"):
    """
    A simulation of

    Args:
        starting_value:
        rate:
        non_decision_time:
        threshold:
        noise:
        time_step_size:
        num_walkers:
        dev:

    Returns:

    """

    particle = torch.full(size=(num_walkers,), fill_value=starting_value, device=dev)
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
rts, decision = simulate_ddms(**ddm_params, num_walkers=NUM_WALKERS, dev=dev)
rts = rts.to("cpu")
decision = decision.to("cpu")
print(f"PyTorch Elapsed: {time.time() - t0}")

#%%

# JIT
jit_simulate_ddms = torch.jit.script(simulate_ddms)

#%%
rts, decision = jit_simulate_ddms(**ddm_params, num_walkers=NUM_WALKERS)

NUM_TIMES = 50
t0 = time.time()
for i in range(NUM_TIMES):
    rts, decision = jit_simulate_ddms(**ddm_params, num_walkers=NUM_WALKERS)
    rts = rts.to("cpu")
    decision = decision.to("cpu")
print(f"JIT Elapsed: {1000 * ((time.time() - t0) / NUM_TIMES)} milliseconds")

#%%
class DriftDiffusionModel(torch.nn.Module):
    def forward(self,
                starting_value,
                rate,
                non_decision_time,
                threshold,
                noise,
                time_step_size,
                num_walkers):
        return simulate_ddms(starting_value, rate, non_decision_time, threshold, noise, time_step_size, num_walkers)

#%%
ddm_model = torch.jit.script(DriftDiffusionModel())

with open('ddm.onnx', 'wb') as file:
    torch.onnx.export(model=ddm_model,
                      args=tuple(ddm_params.values()) + (torch.tensor(NUM_WALKERS),),
                      example_outputs=(rts, decision),
                      f=file,
                      verbose=True,
                      opset_version=11)

#%%
import seaborn as sns
sns.kdeplot(rts)
