import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.objectivefunctions import Distance
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import GridSearch, MINIMIZE
from psyneulink.core.components.functions.nonstateful.transferfunctions import GaussianDistort
from psyneulink.core.components.mechanisms.modulatory.control import OptimizationControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.ports.inputport import SHADOW_INPUTS
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.compositions.composition import Composition, NodeRole
from psyneulink.core.globals.keywords import VARIANCE, NORMED_L0_SIMILARITY
from psyneulink.core.globals.utilities import _SeededPhilox
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism


@pytest.mark.model
@pytest.mark.benchmark(group="Greedy Agent")
def test_simplified_greedy_agent(benchmark, comp_mode):
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
                                 execution_mode=comp_mode)
    assert np.allclose(run_results, [[-200, -108]])
    if benchmark.enabled:
        benchmark(agent_comp.run, **{'inputs':{
            player:[[619,177]],
            prey:[[419,69]],
            }, 'execution_mode':comp_mode})

@pytest.mark.model
@pytest.mark.benchmark(group="Greedy Agant Random")
def test_simplified_greedy_agent_random(benchmark, comp_mode):
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
                                 execution_mode=comp_mode)
    # KDM 12/4/19: modified results due to global seed offset of
    # GaussianDistort assignment.
    # to produce old numbers, run get_global_seed once before creating
    # each Mechanism with GaussianDistort above
    assert np.allclose(run_results, [[-199.5484223217141, -107.79361870517444]])
    if benchmark.enabled:
        benchmark(agent_comp.run, **{'inputs':{
            player:[[619,177]],
            prey:[[419,69]],
            }, 'execution_mode':comp_mode})

@pytest.mark.model
@pytest.mark.benchmark(group="Predator Prey")
@pytest.mark.parametrize("mode", pytest.helpers.get_comp_execution_modes() +
                                 [pytest.helpers.cuda_param('Python-PTX'),
                                  pytest.param('Python-LLVM', marks=pytest.mark.llvm)])
@pytest.mark.parametrize("samples", [[0,10],
    pytest.param([0,3,6,10], marks=pytest.mark.stress),
    pytest.param([0,2,4,6,8,10], marks=pytest.mark.stress),
    pytest.param([a / 10.0 for a in range(0, 101)], marks=pytest.mark.stress),
], ids=lambda x: len(x))
@pytest.mark.parametrize('prng', ['Default', 'Philox'])
@pytest.mark.parametrize('fp_type', [pnl.core.llvm.ir.DoubleType, pnl.core.llvm.ir.FloatType])
def test_predator_prey(benchmark, mode, prng, samples, fp_type):
    if len(samples) > 10 and mode not in {pnl.ExecutionMode.LLVM,
                                          pnl.ExecutionMode.LLVMExec,
                                          pnl.ExecutionMode.LLVMRun,
                                          "Python-PTX", "Python-LLVM"}:
        pytest.skip("This test takes too long")
    if str(mode).startswith('Python-'):
        ocm_mode = mode.split('-')[1]
        mode = pnl.ExecutionMode.Python
    else:
        # OCM default mode is Python
        ocm_mode = 'Python'

    # Instantiate LLVMBuilderContext using the preferred fp type
    pnl.core.llvm.builder_context.LLVMBuilderContext(fp_type())

    benchmark.group = "Predator-Prey " + str(len(samples))
    obs_len = 3
    obs_coords = 2

    player_len = prey_len = predator_len = obs_coords

    # Input Mechanisms
    player_pos = ProcessingMechanism(size=player_len, name="PLAYER POS")
    prey_pos = ProcessingMechanism(size=prey_len, name="PREY POS")
    predator_pos = ProcessingMechanism(size=predator_len, name="PREDATOR POS")

    # Perceptual Mechanisms
    player_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER OBS")
    prey_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY OBS")
    predator_obs = TransferMechanism(size=predator_len, function=GaussianDistort, name="PREDATOR OBS")


    def action_fn(variable):
        predator_pos = variable[0]
        player_pos = variable[1]
        prey_pos = variable[2]

        # Directions away from predator and towards prey
        pred_2_player = player_pos - predator_pos
        play_2_prey = prey_pos - player_pos

        # Distances to predator and prey
        distance_predator = np.sqrt(pred_2_player[0] * pred_2_player[0] + pred_2_player[1] * pred_2_player[1])
        distance_prey = np.sqrt(play_2_prey[0] * play_2_prey[0] + play_2_prey[1] * play_2_prey[1])

        # Normalized directions from predator and towards prey
        pred_2_player_norm = pred_2_player / distance_predator
        play_2_prey_norm = play_2_prey / distance_prey

        # Weighted directions from predator and towards prey
        # weights are reversed so closer agent has greater impact on movement
        pred_2_player_n = pred_2_player_norm * (distance_prey / (distance_predator + distance_prey))
        play_2_prey_n = play_2_prey_norm * (distance_predator / (distance_predator + distance_prey))

        return pred_2_player_n + play_2_prey_n

    # note: unitization is done in main loop
    greedy_action_mech = pnl.ProcessingMechanism(function=action_fn, input_ports=["predator", "player", "prey"],
                                                 default_variable=[[0,0],[0,0],[0,0]], name="ACTION")

    direct_move = ComparatorMechanism(name='DIRECT MOVE',sample=player_pos, target=prey_pos)

    # Create Composition
    agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
    agent_comp.add_linear_processing_pathway([player_pos, player_obs])
    agent_comp.add_linear_processing_pathway([prey_pos, prey_obs])
    agent_comp.add_linear_processing_pathway([predator_pos, predator_obs])
    agent_comp.add_node(greedy_action_mech)
    agent_comp.add_node(direct_move)
    agent_comp.add_projection(pnl.MappingProjection(predator_obs, greedy_action_mech.input_ports[0]))
    agent_comp.add_projection(pnl.MappingProjection(prey_obs, greedy_action_mech.input_ports[1]))
    agent_comp.add_projection(pnl.MappingProjection(player_obs, greedy_action_mech.input_ports[2]))
    agent_comp.exclude_node_roles(direct_move, NodeRole.OUTPUT)


    ocm = OptimizationControlMechanism(state_features=[player_pos, prey_pos, predator_pos],
    # ocm = OptimizationControlMechanism(state_features={SHADOW_INPUTS: [player_pos, prey_pos, predator_pos]},
                                       agent_rep=agent_comp,
                                       function=GridSearch(direction=MINIMIZE,
                                                           save_values=True),

                                       objective_mechanism=ObjectiveMechanism(function=Distance(metric=NORMED_L0_SIMILARITY),
                                                                              monitor=[
                                                                                  greedy_action_mech,
                                                                                  direct_move
                                                                              ]),
                                       control_signals=[ControlSignal(modulates=(VARIANCE, player_obs),
                                                                      allocation_samples=samples,
                                                                      cost_options=pnl.CostFunctions.INTENSITY),
                                                        ControlSignal(modulates=(VARIANCE, predator_obs),
                                                                      allocation_samples=samples,
                                                                      cost_options=pnl.CostFunctions.INTENSITY),
                                                        ControlSignal(modulates=(VARIANCE, prey_obs),
                                                                      allocation_samples=samples,
                                                                      cost_options=pnl.CostFunctions.INTENSITY),
                                                        ],
                                       )
    agent_comp.add_controller(ocm)
    agent_comp.enable_controller = True
    ocm.comp_execution_mode = ocm_mode

    if prng == 'Philox':
        player_obs.function.parameters.random_state.set(_SeededPhilox([0]))
        prey_obs.function.parameters.random_state.set(_SeededPhilox([0]))
        predator_obs.function.parameters.random_state.set(_SeededPhilox([0]))
        ocm.function.parameters.random_state.set(_SeededPhilox([0]))

    input_dict = {player_pos:[[1.1576537,  0.60782117]],
                  predator_pos:[[-0.03479106, -0.47666293]],
                  prey_pos:[[-0.60836214,  0.1760381 ]],
                 }
    run_results = agent_comp.run(inputs=input_dict, num_trials=2, execution_mode=mode)

    if len(samples) == 2:
        if prng == 'Default':
            assert np.allclose(run_results[0], [[0.9705216285127504, -0.1343332460369043]])
        elif prng == 'Philox':
            if mode == pnl.ExecutionMode.Python or pytest.helpers.llvm_current_fp_precision() == 'fp64':
                assert np.allclose(run_results[0], [[-0.16882940384606543, -0.07280074899749223]])
            elif pytest.helpers.llvm_current_fp_precision() == 'fp32':
                assert np.allclose(run_results[0], [[-0.8639436960220337, 0.4983368515968323]])
            else:
                assert False, "Unkown FP type!"
        else:
            assert False, "Unknown PRNG!"

        if mode == pnl.ExecutionMode.Python:
            # FIXEM: The results are 'close' for both Philox and MT,
            #        because they're dominated by costs
            assert np.allclose(np.asfarray(ocm.function.saved_values).flatten(),
                               [-2.66258741, -22027.9970321, -22028.17515945, -44053.59867802,
                                -22028.06045185, -44053.4048842, -44053.40736234, -66078.90687915])

    if benchmark.enabled:
        benchmark(agent_comp.run, inputs=input_dict, execution_mode=mode)
