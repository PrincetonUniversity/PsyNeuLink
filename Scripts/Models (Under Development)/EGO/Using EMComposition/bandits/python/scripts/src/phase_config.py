from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PhaseConfig:
    """
    Configuration for a single phase of the task.

    A phase defines two possible state sequences, the number of times a
    sequence is sampled, and the reward associated with each state. The
    generator will randomly choose between `state_seq_1` and `state_seq_2`
    for each sequence.

    Attributes
    ----------
    state_seq_1 : List[int]
        First possible state sequence for this phase.

    state_seq_2 : List[int]
        Second possible state sequence for this phase.

    num_sequences : int
        Number of sequences (trials) to generate in this phase.

    reward_by_state : Dict[int, float]
        Mapping from state index to reward. Every state appearing in the
        sequences must have a corresponding reward defined.
    """
    state_seq_1: List[int]
    state_seq_2: List[int]
    num_sequences: int
    reward_by_state: Dict[int, float]

    def __post_init__(self):
        """
        Minimal check to assert each state has a reward.
        """

        states = set(self.state_seq_1 + self.state_seq_2)
        reward_keys = set(self.reward_by_state.keys())

        missing = states - reward_keys
        if missing:
            raise ValueError(f'Missing reward keys: {missing}')
