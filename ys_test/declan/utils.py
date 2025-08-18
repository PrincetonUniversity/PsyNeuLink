import random

import numpy as np
import torch


def set_random_seed_and_capture_state(params):
    try:
        seed = params.seed
    except AttributeError:
        seed = params

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)

    # Capture RNG states
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            state["torch_mps"] = torch.mps.get_rng_state()
        except Exception:
            pass  # not always supported

    return state

def compare_rng_states(state1, state2, name):
    print(f"\n🔍 Comparing RNG states: {name}")
    for key in state1:
        if key not in state2:
            print(f"  ❌ Key '{key}' missing in second state")
            continue

        v1 = state1[key]
        v2 = state2[key]

        if isinstance(v1, tuple):
            # Special case: random.getstate() or np.get_state()
            if len(v1) != len(v2):
                print(f"  ❌ {key} state length mismatch")
                continue

            match = True
            for i, (a, b) in enumerate(zip(v1, v2)):
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    if not np.array_equal(a, b):
                        print(f"  ❌ {key}[{i}] arrays differ")
                        match = False
                        break
                else:
                    if a != b:
                        print(f"  ❌ {key}[{i}] differs: {a} vs {b}")
                        match = False
                        break

            if match:
                print(f"  ✅ {key} state matches")
        elif isinstance(v1, torch.Tensor):
            if not torch.equal(v1, v2):
                print(f"  ❌ {key} tensor state differs")
            else:
                print(f"  ✅ {key} tensor state matches")
        else:
            if v1 != v2:
                print(f"  ❌ {key} value differs: {v1} vs {v2}")
            else:
                print(f"  ✅ {key} matches")
