import numpy as np
import pytest

import psyneulink as pnl

# This script implements Figure 1 of Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001).
# Conflict monitoring and cognitive control. Psychological Review, 108, 624–652.
# http://dx.doi.org/10.1037/0033-295X.108.3.624

# Figure 1 plots the ENERGY computed by a conflict mechanism. It is highest for incongruent trials,
# and similar for congruent and neutral trials.
# Noise is turned of and for each condition we ran one trial only. A response threshold was not defined. Responses were
# made at the marked * signs in the figure.
# Note that this script implements a slightly different Figure than in the original Figure in the paper.
# However, this implementation is identical with a plot we created with an old MATLAB code which was used for the
# conflict monitoring simulations.
import psyneulink.core.components.functions.objectivefunctions
import psyneulink.core.components.functions.transferfunctions


@pytest.mark.model
@pytest.mark.benchmark
@pytest.mark.parametrize("reps", [1,
                                  pytest.param(10, marks=pytest.mark.stress),
                                  pytest.param(100, marks=pytest.mark.stress)])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                  pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                  pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_botvinick_model(benchmark, mode, reps):
    benchmark.group = "Botvinick (scale " + str(reps / 100) + ")"

    # SET UP MECHANISMS ----------------------------------------------------------------------------------------------------
    # Linear input layer
    # colors: ('red', 'green'), words: ('RED','GREEN')
    colors_input_layer = pnl.TransferMechanism(size=3,
                                               function=psyneulink.core.components.Linear,
                                               name='COLORS_INPUT')

    words_input_layer = pnl.TransferMechanism(size=3,
                                              function=psyneulink.core.components.Linear,
                                              name='WORDS_INPUT')

    task_input_layer = pnl.TransferMechanism(size=2,
                                             function=psyneulink.core.components.Linear,
                                             name='TASK_INPUT')

    #   Task layer, tasks: ('name the color', 'read the word')
    task_layer = pnl.RecurrentTransferMechanism(size=2,
                                                function=psyneulink.core.components.Logistic,
                                                hetero=-2,
                                                integrator_mode=True,
                                                integration_rate=0.01,
                                                name='TASK_LAYER')

    # Hidden layer
    # colors: ('red','green', 'neutral') words: ('RED','GREEN', 'NEUTRAL')
    colors_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                         function=psyneulink.core.components.Logistic(x_0=4.0),  # bias 4.0 is -4.0 in the paper see Docs for description
                                                         integrator_mode=True,
                                                         hetero=-2,
                                                         integration_rate=0.01,  # cohen-huston text says 0.01
                                                         name='COLORS_HIDDEN')

    words_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                        function=psyneulink.core.components.Logistic(x_0=4.0),
                                                        integrator_mode=True,
                                                        hetero=-2,
                                                        integration_rate=0.01,
                                                        name='WORDS_HIDDEN')

    #   Response layer, responses: ('red', 'green')
    response_layer = pnl.RecurrentTransferMechanism(size=2,
                                                    function=psyneulink.core.components.Logistic,
                                                    hetero=-2.0,
                                                    integrator_mode=True,
                                                    integration_rate=0.01,
                                                    output_ports = [pnl.RESULT,
                                                                     {pnl.NAME: 'DECISION_ENERGY',
                                                                      pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                                      pnl.FUNCTION: psyneulink.core.components.Stability(
                                                                          default_variable = np.array([0.0, 0.0]),
                                                                          metric = pnl.ENERGY,
                                                                          matrix = np.array([[0.0, -4.0],
                                                                                            [-4.0, 0.0]]))}],
                                                    name='RESPONSE', )

    # Mapping projections---------------------------------------------------------------------------------------------------

    color_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                                 [0.0, 1.0, 0.0],
                                                                 [0.0, 0.0, 1.0]]))

    word_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                                [0.0, 1.0, 0.0],
                                                                [0.0, 0.0, 1.0]]))

    task_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0],
                                                                [0.0, 1.0]]))

    color_task_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 0.0],
                                                                 [4.0, 0.0],
                                                                 [4.0, 0.0]]))

    task_color_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 4.0, 4.0],
                                                                 [0.0, 0.0, 0.0]]))

    response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                    [0.0, 1.5, 0.0]]))

    response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                    [0.0, 2.5, 0.0]]))

    color_response_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0],
                                                                    [0.0, 1.5],
                                                                    [0.0, 0.0]]))

    word_response_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0],
                                                                    [0.0, 2.5],
                                                                    [0.0, 0.0]]))

    word_task_weights = pnl.MappingProjection(matrix=np.array([[0.0, 4.0],
                                                               [0.0, 4.0],
                                                               [0.0, 4.0]]))

    task_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                               [4.0, 4.0, 4.0]]))

    # CREATE Composition
    comp = pnl.Composition()

    # Add mechanisms
    comp.add_node(colors_input_layer)
    comp.add_node(colors_hidden_layer)

    comp.add_node(words_input_layer)
    comp.add_node(words_hidden_layer)

    comp.add_node(task_input_layer)
    comp.add_node(task_layer)
    comp.add_node(response_layer)

    # Add projections
    comp.add_projection(task_input_weights, task_input_layer, task_layer)

    # Color process
    comp.add_projection(color_input_weights, colors_input_layer, colors_hidden_layer)
    comp.add_projection(color_response_weights, colors_hidden_layer, response_layer)
    comp.add_projection(response_color_weights, response_layer, colors_hidden_layer)

    # Word process
    comp.add_projection(word_input_weights, words_input_layer, words_hidden_layer)
    comp.add_projection(word_response_weights, words_hidden_layer, response_layer)
    comp.add_projection(response_word_weights, response_layer, words_hidden_layer)

    # Color task process
    comp.add_projection(task_color_weights, task_layer, colors_hidden_layer)
    comp.add_projection(color_task_weights, colors_hidden_layer, task_layer)

    # Word task process
    comp.add_projection(task_word_weights, task_layer, words_hidden_layer)
    comp.add_projection(word_task_weights, words_hidden_layer, task_layer)

    def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):
        trialdict = {
        colors_input_layer: [red_color, green_color, neutral_color],
        words_input_layer: [red_word, green_word, neutral_word],
        task_input_layer: [CN, WR]
        }
        return trialdict

    # Define initialization trials separately
    CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)  #red_color, green color, red_word, green word, CN, WR
    CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0) #red_color, green color, red_word, green word, CN, WR
    CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0)   #red_color, green color, red_word, green word, CN, WR
    CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 1, 1, 0)     #red_color, green color, red_word, green word, CN, WR

    Stimulus = [[CN_trial_initialize_input, CN_congruent_trial_input],
                [CN_trial_initialize_input, CN_incongruent_trial_input],
                [CN_trial_initialize_input, CN_control_trial_input]]

    # should be 500 and 1000
    ntrials0 = 5 * reps
    ntrials = 10 * reps

    def run(bin_execute):
        results = []
        for i, stim in enumerate(Stimulus):
            # RUN the COMPOSITION to initialize --------------------------------
            exec_id = "exec_" + str(i)
            comp.run(inputs=stim[0], num_trials=ntrials0, bin_execute=bin_execute, context=exec_id)
            comp.run(inputs=stim[1], num_trials=ntrials, bin_execute=bin_execute, context=exec_id)

            # Comp results include concatenation of both the above runs
            results.append(comp.results)

        return results

    res = run(mode)
    # the corresponding output port indices in composition results
    # these were 0 and 1 in the prior version of the test
    response_results_index = 3
    response_decision_energy_index = 4
    if reps == 1:
        res2d = [x[response_results_index] for r in res for x in r]
        # NOTE: The formatting below provides visual split between
        #       initialization and runs. Please do not change it.
        assert np.allclose(res2d, [[0.497679878752004, 0.497679878752004],
                                   [0.4954000154631831, 0.4954000154631831],
                                   [0.4931599760310996, 0.4931599760310996],
                                   [0.4909593232354856, 0.4909593232354856],
                                   [0.4887976172454234, 0.4887976172454234],
[0.4866744160981826, 0.4866744160981826],
[0.4845913708928323, 0.4845892761509014],
[0.4825481248817388, 0.4825417411478152],
[0.4805443207349158, 0.4805313535747981],
[0.4785796008355799, 0.4785576550393446],
[0.4766536075535322, 0.4766201866281457],
[0.4747659834976122, 0.4747184892433102],
[0.4729163717484405, 0.4728521039182508],
[0.471104416072617, 0.4710205721142167],
[0.4693297611195044, 0.4692234359984253],
                                   [0.497679878752004, 0.497679878752004],
                                   [0.4954000154631831, 0.4954000154631831],
                                   [0.4931599760310996, 0.4931599760310996],
                                   [0.4909593232354856, 0.4909593232354856],
                                   [0.4887976172454234, 0.4887976172454234],
[0.4866744160981826, 0.4866744160981826],
[0.4845900488307297, 0.484590598212878],
[0.4825440921072959, 0.4825457739209331],
[0.4805361215634117, 0.4805395527400902],
[0.478565712169312, 0.4785715436856292],
[0.4766324385709689, 0.4766413555592825],
[0.4747358754098249, 0.4747485972170732],
[0.4728755976222633, 0.4728928778172651],
[0.4710511807198248, 0.4710738070495766],
[0.4692622010511354, 0.4692909953467661],
                                   [0.497679878752004, 0.497679878752004],
                                   [0.4954000154631831, 0.4954000154631831],
                                   [0.4931599760310996, 0.4931599760310996],
                                   [0.4909593232354856, 0.4909593232354856],
                                   [0.4887976172454234, 0.4887976172454234],
[0.4866744160981826, 0.4866744160981826],
[0.4845900488307297, 0.4845892761509014],
[0.4825440987479238, 0.4825417477884414],
[0.4805361483673633, 0.480531380378737],
[0.4785657797808562, 0.4785577226508331],
[0.4766325749933464, 0.476620323050344],
[0.4747361162403673, 0.4747187300733827],
[0.4728759862850626, 0.4728524925799762],
[0.4710517686957791, 0.4710211600879554],
[0.469263048105203, 0.469224283048266]])

        res1d = [x[response_decision_energy_index] for r in res for x in r]
        assert np.allclose(res1d, [[0.9907410468584376], [0.9816847012836883], [0.9728270478359791], [0.964164228287384], [0.9556924424992138],
[0.9474079491380278], [0.9393111265997224], [0.9313984494721904], [0.9236664515817239], [0.9161117261021626],
[0.9087309255565738], [0.9015207617204032], [0.8944780054345429], [0.8875994863362322], [0.880882092515256],
                                   [0.9907410468584376], [0.9816847012836883], [0.9728270478359791], [0.964164228287384], [0.9556924424992138],
[0.9474079491380278], [0.9393111266035642], [0.9313984495075565], [0.9236664517261579], [0.9161117265115205],
[0.9087309264959722], [0.9015207635977346], [0.8944780088366047], [0.887599492067544], [0.8808821016396065],
                                   [0.9907410468584376], [0.9816847012836883], [0.9728270478359791], [0.964164228287384], [0.9556924424992138],
[0.9474079491380278], [0.939308563971253], [0.9313906911792855], [0.9236507947874026], [0.9160853988421866],
[0.9086910874785843], [0.901464504886388], [0.894402355184426], [0.8875014022102764], [0.8807584692328312]])
    if reps == 10:
        assert np.allclose(res[0][ntrials0 - 1][response_results_index], [0.42481045, 0.42481045])
        assert np.allclose(res[0][-1][response_results_index], [0.43512335, 0.39995991])
        assert np.allclose(res[1][ntrials0 - 1][response_results_index], [0.42481045, 0.42481045])
        assert np.allclose(res[1][-1][response_results_index], [0.41360321, 0.42121262])
        assert np.allclose(res[2][ntrials0 - 1][response_results_index], [0.42481045, 0.42481045])
        assert np.allclose(res[2][-1][response_results_index], [0.41621778, 0.40255998])
        assert np.allclose(res[0][ntrials0 - 1][response_decision_energy_index], [0.72185566])
        assert np.allclose(res[0][-1][response_decision_energy_index], [0.69612758])
        assert np.allclose(res[1][ntrials0 - 1][response_decision_energy_index], [0.72185566])
        assert np.allclose(res[1][-1][response_decision_energy_index], [0.69685957])
        assert np.allclose(res[2][ntrials0 - 1][response_decision_energy_index], [0.72185566])
        assert np.allclose(res[2][-1][response_decision_energy_index], [0.67021047])
    if reps == 100:
        assert np.allclose(res[0][ntrials0 - 1][response_results_index], [0.48590224, 0.48590224])
        assert np.allclose(res[0][-1][response_results_index], [0.95967791, 0.21434208])
        assert np.allclose(res[1][ntrials0 - 1][response_results_index], [0.48590224, 0.48590224])
        assert np.allclose(res[1][-1][response_results_index], [0.55847666, 0.83814112])
        assert np.allclose(res[2][ntrials0 - 1][response_results_index], [0.48590224, 0.48590224])
        assert np.allclose(res[2][-1][response_results_index], [0.89673726, 0.25100269])
        assert np.allclose(res[0][ntrials0 - 1][response_decision_energy_index], [0.94440397])
        assert np.allclose(res[0][-1][response_decision_energy_index], [0.82279743])
        assert np.allclose(res[1][ntrials0 - 1][response_decision_energy_index], [0.94440397])
        assert np.allclose(res[1][-1][response_decision_energy_index], [1.87232903])
        assert np.allclose(res[2][ntrials0 - 1][response_decision_energy_index], [0.94440397])
        assert np.allclose(res[2][-1][response_decision_energy_index], [0.90033387])
    if benchmark.enabled:
        benchmark(run, mode)
