import pytest
import numpy as np
import psyneulink as pnl

from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.functions.transferfunctions import GaussianDistort
from psyneulink.core.compositions.composition import Composition

@pytest.mark.model
@pytest.mark.benchmark
@pytest.mark.parametrize("mode", ['Python',
    pytest.param('LLVM', marks=[pytest.mark.llvm]),
    pytest.param('LLVMExec', marks=[pytest.mark.llvm]),
    pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
    pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
    pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_simplified_greedy_agent(benchmark, mode):
    # These should probably be replaced by reference to ForagerEnv constants:
    obs_len = 2
    action_len = 2
    player_coord_idx = slice(0,2)
    predator_coord_idx = slice(3,5)
    prey_coord_idx = slice(6,8)
    player_value_idx = 2
    predator_value_idx = 5
    prey_value_idx = 8

    player_len = prey_len = predator_len = obs_len

# The original needs GaussianDistort
#    player = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER OBS")
#    prey = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY OBS")
    player = TransferMechanism(size=prey_len, name="PLAYER OBS")
    prey = TransferMechanism(size=prey_len, name="PREY OBS")

    # For future use:
    values = TransferMechanism(size=3, name="AGENT VALUES")
    reward = TransferMechanism(name="REWARD")

    # Use ComparatorMechanism to compute direction of action as difference of coordinates between player and prey:
    # note: unitization is done in main loop, to allow compilation of LinearCombination function) (TBI)
    greedy_action_mech = ComparatorMechanism(name='MOTOR OUTPUT',sample=player,target=prey)

    agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
    agent_comp.add_node(player)
    agent_comp.add_node(prey)
    agent_comp.add_node(greedy_action_mech)

    # Projections to greedy_action_mech were created by assignments of sample and target args in its constructor,
    #  so just add them to the Composition).
    for projection in greedy_action_mech.projections:
        agent_comp.add_projection(projection)

    run_results = agent_comp.run(inputs={player:[[619,177]],
                                         prey:[[419,69]]},
                                 bin_execute=mode)
    assert np.allclose(run_results, [[-200, -108]])
    benchmark(agent_comp.run, **{'inputs':{
        player:[[619,177]],
        prey:[[419,69]],
        }, 'bin_execute':mode})
