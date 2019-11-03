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
    if reps == 1:
        res2d = [x[0] for r in res for x in r]
        # NOTE: The formatting below provides visual split between
        #       initialization and runs. Please do not change it.
        assert np.allclose(res2d, [[0.4976852381289525, 0.4976852381289525],
                                   [0.4954107346393883, 0.4954107346393883],
                                   [0.493176053709877,  0.493176053709877],
                                   [0.49098075641903416, 0.49098075641903416],
                                   [0.48882440125362586, 0.48882440125362586],
[0.4867086398433437, 0.4867065445883512],
[0.48463311528336894, 0.4846267297743475],
[0.48259746969966644, 0.4825844985226081],
[0.4806013445692439, 0.4805793913088515],
[0.4786443810171789, 0.478610947754902],
[0.47672622009168814, 0.47667870698765635],
[0.4748465030184808, 0.4747822079766836],
[0.47300487143558895, 0.47292098985146575],
[0.4712009676098337, 0.47109459219926386],
[0.4694344346360397, 0.469302555344549],
                                   [0.4976852381289525, 0.4976852381289525],
                                   [0.4954107346393883, 0.4954107346393883],
                                   [0.493176053709877, 0.493176053709877],
                                   [0.49098075641903416, 0.49098075641903416],
                                   [0.48882440125362586, 0.48882440125362586],
[0.4867073174570701, 0.48670786697451607],
[0.4846290813861633, 0.4846307636703061],
[0.4825892679847457, 0.4825927002315042],
[0.4805874511719694, 0.4805932846864651],
[0.4786232042071176, 0.47863212451395776],
[0.4766961000320741, 0.47670882693361377],
[0.47480571159250756, 0.47482299917547827],
[0.47295161213880393, 0.4729742487298337],
[0.47113337550774326, 0.47116218357843975],
[0.46935057638588656, 0.4693864124082814],
                                   [0.4976852381289525, 0.4976852381289525],
                                   [0.4954107346393883, 0.4954107346393883],
                                   [0.493176053709877, 0.493176053709877],
                                   [0.49098075641903416, 0.49098075641903416],
                                   [0.48882440125362586, 0.48882440125362586],
[0.4867073174570701, 0.4867065445883512],
[0.484629088030232, 0.4846267364184149],
[0.4825892948040597, 0.4825845253419109],
[0.4805875188258231, 0.48057945896265514],
[0.4786233407217324, 0.4786110842693559],
[0.47669634103703906, 0.4766789479921973],
[0.4748061005546799, 0.4747825969378818],
[0.47295220059350046, 0.47292157830414155],
[0.471134223287054, 0.47109543997470166],
[0.4693517518439444, 0.4693037307956372]])
        res1d = [x[1] for r in res for x in r]
        assert np.allclose(res1d, [[0.9907623850058885], [0.9817271839837536], [0.9728904798113899], [0.9642484126952278], [0.9557971810438632],
[0.9475371212778005], [0.9394646472005338], [0.9315762316131724], [0.9238684065412114], [0.9163377633447616],
[0.9089809527216751], [0.901794684612485],  [0.8947757280155361], [0.8879209107202126], [0.8812271189656683],
                                   [0.9907623850058885], [0.9817271839837536], [0.9728904798113899], [0.9642484126952278], [0.9557971810438632],
[0.9475371212816769], [0.9394646472360609], [0.9315762317580137], [0.9238684069513318], [0.9163377642853222],
[0.9089809546004746], [0.9017946880160064], [0.894775733747658],  [0.8879209198436373], [0.8812271328461213],
                                   [0.9907623850058885], [0.9817271839837536], [0.9728904798113899], [0.9642484126952278], [0.9557971810438632],
[0.9475345468215851], [0.9394568532220963], [0.9315605030724186], [0.9238419591260757], [0.9162977442377989],
[0.9089244414290619], [0.9017186938531998], [0.8946772046683807], [0.8877967368262162], [0.8810741127833248]])
    if reps == 10:
        assert np.allclose(res[0][ntrials0 - 1][0], [0.42505118, 0.42505118])
        assert np.allclose(res[0][-1][0], [0.43621363, 0.40023224])
        assert np.allclose(res[1][ntrials0 - 1][0], [0.42505118, 0.42505118])
        assert np.allclose(res[1][-1][0], [0.41420086, 0.42196304])
        assert np.allclose(res[2][ntrials0 - 1][0], [0.42505118, 0.42505118])
        assert np.allclose(res[2][-1][0], [0.41689666, 0.40291293])
        assert np.allclose(res[0][ntrials0 - 1][1], [0.72267401])
        assert np.allclose(res[0][-1][1], [0.69834703])
        assert np.allclose(res[1][ntrials0 - 1][1], [0.72267401])
        assert np.allclose(res[1][-1][1], [0.69910981])
        assert np.allclose(res[2][ntrials0 - 1][1], [0.72267401])
        assert np.allclose(res[2][-1][1], [0.67189222])
    if reps == 100:
        assert np.allclose(res[0][ntrials0 - 1][0], [0.48611807, 0.48611807])
        assert np.allclose(res[0][-1][0], [0.95970536, 0.21425063])
        assert np.allclose(res[1][ntrials0 - 1][0], [0.48611807, 0.48611807])
        assert np.allclose(res[1][-1][0], [0.55802971, 0.83844741])
        assert np.allclose(res[2][ntrials0 - 1][0], [0.48611807, 0.48611807])
        assert np.allclose(res[2][-1][0], [0.89746087, 0.25060644])
        assert np.allclose(res[0][ntrials0 - 1][1], [0.94524311])
        assert np.allclose(res[0][-1][1], [0.82246989])
        assert np.allclose(res[1][ntrials0 - 1][1], [0.94524311])
        assert np.allclose(res[1][-1][1], [1.87151424])
        assert np.allclose(res[2][ntrials0 - 1][1], [0.94524311])
        assert np.allclose(res[2][-1][1], [0.89963791])

    benchmark(run, mode)
