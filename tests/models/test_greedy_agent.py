import pytest
import numpy as np

from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.modulatory.control import OptimizationControlMechanism
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.components.functions.optimizationfunctions import GridSearch, MINIMIZE
from psyneulink.core.components.functions.transferfunctions import GaussianDistort, Exponential
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.inputport import SHADOW_INPUTS
from psyneulink.core.compositions.composition import Composition, NodeRole
from psyneulink.core.globals.keywords import VARIANCE, NORMED_L0_SIMILARITY

@pytest.mark.model
@pytest.mark.benchmark(group="Greedy Agent")
@pytest.mark.parametrize("mode", ['Python',
    pytest.param('LLVM', marks=[pytest.mark.llvm]),
    pytest.param('LLVMExec', marks=[pytest.mark.llvm]),
    pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
    pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
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

@pytest.mark.model
@pytest.mark.benchmark(group="Greedy Agant Random")
@pytest.mark.parametrize("mode", ['Python',
    pytest.param('LLVM', marks=[pytest.mark.llvm]),
    pytest.param('LLVMExec', marks=[pytest.mark.llvm]),
    pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
    pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
    pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_simplified_greedy_agent_random(benchmark, mode):
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

    player = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER OBS")
    prey = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY OBS")

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
    # KDM 12/4/19: modified results due to global seed offset of
    # GaussianDistort assignment.
    # to produce old numbers, run get_global_seed once before creating
    # each Mechanism with GaussianDistort above
    assert np.allclose(run_results, [[-199.5484223217141, -107.79361870517444]])
    benchmark(agent_comp.run, **{'inputs':{
        player:[[619,177]],
        prey:[[419,69]],
        }, 'bin_execute':mode})

@pytest.mark.model
@pytest.mark.benchmark(group="Predator Prey")
@pytest.mark.parametrize("mode", ['Python',
     pytest.param('Python-PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
     pytest.param('LLVM', marks=[pytest.mark.llvm]),
     pytest.param('LLVMExec', marks=[pytest.mark.llvm]),
     pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
     pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
     pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
])
@pytest.mark.parametrize("samples", [[0,10],
    pytest.param([a / 10.0 for a in range(0, 101)]),
    pytest.param([0,3,6,10], marks=pytest.mark.stress),
    pytest.param([0,2,4,6,8,10], marks=pytest.mark.stress),
], ids=lambda x: len(x))
def test_predator_prey(benchmark, mode, samples):
    if len(samples) > 10 and mode not in {"LLVMRun", "Python-PTX"}:
        pytest.skip("This test takes too long")
    # OCM default mode is Python
    mode, ocm_mode = (mode + "-Python").split('-')[0:2]
    benchmark.group = "Predator-Prey " + str(len(samples))
    obs_len = 3
    obs_coords = 2
    player_idx = 0
    player_obs_start_idx = player_idx * obs_len
    player_value_idx = player_idx * obs_len + obs_coords
    player_coord_slice = slice(player_obs_start_idx,player_value_idx)
    predator_idx = 1
    predator_obs_start_idx = predator_idx * obs_len
    predator_value_idx = predator_idx * obs_len + obs_coords
    predator_coord_slice = slice(predator_obs_start_idx,predator_value_idx)
    prey_idx = 2
    prey_obs_start_idx = prey_idx * obs_len
    prey_value_idx = prey_idx * obs_len + obs_coords
    prey_coord_slice = slice(prey_obs_start_idx,prey_value_idx)

    player_len = prey_len = predator_len = obs_coords

    # Perceptual Mechanisms
    player_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER OBS")
    prey_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY OBS")
    predator_obs = TransferMechanism(size=predator_len, function=GaussianDistort, name="PREDATOR OBS")

    # Action Mechanism
    #    Use ComparatorMechanism to compute direction of action as difference of coordinates between player and prey:
    #    note: unitization is done in main loop, to allow compilation of LinearCombination function) (TBI)
    greedy_action_mech = ComparatorMechanism(name='ACTION',sample=player_obs,target=prey_obs)

    # Create Composition
    agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
    agent_comp.add_node(player_obs)
    agent_comp.add_node(predator_obs)
    agent_comp.add_node(prey_obs)
    agent_comp.add_node(greedy_action_mech)
    agent_comp.exclude_node_roles(predator_obs, NodeRole.OUTPUT)

    ocm = OptimizationControlMechanism(features={SHADOW_INPUTS: [player_obs, predator_obs, prey_obs]},
                                       agent_rep=agent_comp,
                                       function=GridSearch(direction=MINIMIZE,
                                                           save_values=True),

                                       objective_mechanism=ObjectiveMechanism(function=Distance(metric=NORMED_L0_SIMILARITY),
                                                                              monitor=[
                                                                                  player_obs,
                                                                                  prey_obs
                                                                              ]),
                                       control_signals=[ControlSignal(modulates=(VARIANCE,player_obs),
                                                                      allocation_samples=samples),
                                                        ControlSignal(modulates=(VARIANCE,predator_obs),
                                                                      allocation_samples=samples),
                                                        ControlSignal(modulates=(VARIANCE,prey_obs),
                                                                      allocation_samples=samples)
                                                        ],
                                       )
    agent_comp.add_controller(ocm)
    agent_comp.enable_controller = True
    ocm.comp_execution_mode = ocm_mode

    input_dict = {player_obs:[[1.1576537,  0.60782117]],
                  predator_obs:[[-0.03479106, -0.47666293]],
                  prey_obs:[[-0.60836214,  0.1760381 ]],
                 }
    run_results = agent_comp.run(inputs=input_dict, num_trials=2, bin_execute=mode)

    if len(samples) == 2:
        # KDM 12/4/19: modified results due to global seed offset of
        # GaussianDistort assignment.
        # to produce old numbers, run get_global_seed once before creating
        # each Mechanism with GaussianDistort above
        assert np.allclose(run_results[0], [[-10.06333025,   2.4845505 ]])
        if mode == 'Python':
            assert np.allclose(ocm.feature_values, [[ 1.1576537,   0.60782117],
                                                    [-0.03479106, -0.47666293],
                                                    [-0.60836214,  0.1760381 ]])

    benchmark(agent_comp.run, inputs=input_dict, bin_execute=mode)
