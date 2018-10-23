import numpy as np
#import matplotlib.pyplot as plt
import psyneulink as pnl
import pytest

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

@pytest.mark.model
@pytest.mark.benchmark
@pytest.mark.parametrize("reps", [1, 10, 100])
@pytest.mark.parametrize("mode", ['Python', 'LLVM', 'LLVMExec', 'LLVMRun'])
def test_botvinick_model(benchmark, mode, reps):
    if reps > 1 and not pytest.config.getoption("--stress"):
        benchmark.disabled = True
        benchmark(lambda _:0,0)
        pytest.skip("not stressed")
        return # This should not be reached

    benchmark.group = "Botvinick (scale " + str(reps/100) + ")";

    # SET UP MECHANISMS ----------------------------------------------------------------------------------------------------
    # Linear input layer
    # colors: ('red', 'green'), words: ('RED','GREEN')
    colors_input_layer = pnl.TransferMechanism(size=3,
                                               function=pnl.Linear,
                                               name='COLORS_INPUT')

    words_input_layer = pnl.TransferMechanism(size=3,
                                              function=pnl.Linear,
                                              name='WORDS_INPUT')

    task_input_layer = pnl.TransferMechanism(size=2,
                                              function=pnl.Linear,
                                              name='TASK_INPUT')

    #   Task layer, tasks: ('name the color', 'read the word')
    task_layer = pnl.RecurrentTransferMechanism(size=2,
                                                function=pnl.Logistic(),
                                                hetero=-2,
                                                integrator_mode=True,
                                                integration_rate=0.01,
                                                name='TASK_LAYER')

    # Hidden layer
    # colors: ('red','green', 'neutral') words: ('RED','GREEN', 'NEUTRAL')
    colors_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                         function=pnl.Logistic(bias=4.0), # bias 4.0 is -4.0 in the paper see Docs for description
                                                         integrator_mode=True,
                                                         hetero=-2,
                                                         integration_rate=0.01, # cohen-huston text says 0.01
                                                         name='COLORS_HIDDEN')

    words_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                        function=pnl.Logistic(bias=4.0),
                                                        integrator_mode=True,
                                                        hetero=-2,
                                                        integration_rate=0.01,
                                                        name='WORDS_HIDDEN')

    #   Response layer, responses: ('red', 'green')
    response_layer = pnl.RecurrentTransferMechanism(size=2,
                                                    function=pnl.Logistic(),
                                                    hetero=-2.0,
                                                    integrator_mode=True,
                                                    integration_rate=0.01,
                                                    output_states = [pnl.RECURRENT_OUTPUT.RESULT,
                                                                     {pnl.NAME: 'DECISION_ENERGY',
                                                                      pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                                      pnl.FUNCTION: pnl.Stability(
                                                                          default_variable = np.array([0.0, 0.0]),
                                                                          metric = pnl.ENERGY,
                                                                          matrix = np.array([[0.0, -4.0],
                                                                                            [-4.0, 0.0]]))}],
                                                    name='RESPONSE',)

    # Log ------------------------------------------------------------------------------------------------------------------
    response_layer.set_log_conditions('DECISION_ENERGY')

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
    comp.add_c_node(colors_input_layer)
    comp.add_c_node(colors_hidden_layer)

    comp.add_c_node(words_input_layer)
    comp.add_c_node(words_hidden_layer)

    comp.add_c_node(task_input_layer)
    comp.add_c_node(task_layer)
    comp.add_c_node(response_layer)

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
    comp._analyze_graph()

    def run(bin_execute):
        results = []
        for stim in Stimulus:
        # RUN the SYSTEM to initialize ----------------------------------------------------------------------------------------
            comp.run(inputs=stim[0], num_trials=ntrials0, bin_execute=bin_execute)
            comp.run(inputs=stim[1], num_trials=ntrials, bin_execute=bin_execute)
            # reinitialize after condition was run
            colors_hidden_layer.reinitialize([[0,0,0]])
            words_hidden_layer.reinitialize([[0,0,0]])
            response_layer.reinitialize([[0,0]])
            task_layer.reinitialize([[0,0]])
            # Comp results include concatenation of both the above runs
            results.append(comp.results.copy())
            comp.reinitialize()
            comp.results = []

        return results

    res = benchmark(run, mode)
    if reps == 1:
        res2d = [x[0] for r in res for x in r]
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
        # FIXME: for some reason numpy adds another layer of array
        if mode == 'Python':
            res1d = [x[1][0] for r in res for x in r]
        else:
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

    if mode[:4] == 'LLVM' or reps != 10:
        return
    r2 = response_layer.log.nparray_dictionary('DECISION_ENERGY') #get logged DECISION_ENERGY dictionary
    energy = r2['DECISION_ENERGY']                                #save logged DECISION_ENERGY

    assert np.allclose(energy[:450],
[ 0.9907482,  0.98169891, 0.97284822, 0.96419228, 0.95572727, 0.94744946,
  0.93935517, 0.93144078, 0.92370273, 0.91613752, 0.90874171, 0.90151191,
  0.89444481, 0.88753715, 0.88078573, 0.8741874,  0.8677391,  0.86143779,
  0.85528051, 0.84926435, 0.84338646, 0.83764405, 0.83203438, 0.82655476,
  0.82120257, 0.81597521, 0.81087018, 0.805885,   0.80101724, 0.79626453,
  0.79162455, 0.78709503, 0.78267373, 0.77835847, 0.77414712, 0.77003759,
  0.76602783, 0.76211584, 0.75829965, 0.75457736, 0.75094707, 0.74740696,
  0.74395521, 0.74059008, 0.73730983, 0.73411278, 0.73099728, 0.72796172,
  0.7250045,  0.72212409, 0.71932195, 0.71660193, 0.7139626,  0.71140257,
  0.70892047, 0.70651498, 0.70418478, 0.70192859, 0.69974515, 0.69763324,
  0.69559166, 0.69361922, 0.69171476, 0.68987717, 0.68810533, 0.68639816,
  0.68475458, 0.68317357, 0.6816541,  0.68019517, 0.67879579, 0.67745502,
  0.6761719,  0.67494551, 0.67377494, 0.67265932, 0.67159776, 0.67058942,
  0.66963346, 0.66872906, 0.66787541, 0.66707173, 0.66631723, 0.66561116,
  0.66495278, 0.66434134, 0.66377614, 0.66325648, 0.66278164, 0.66235097,
  0.66196379, 0.66161945, 0.66131731, 0.66105673, 0.66083709, 0.6606578,
  0.66051824, 0.66041784, 0.66035601, 0.66033219, 0.66034583, 0.66039637,
  0.66048328, 0.66060603, 0.6607641,  0.66095698, 0.66118416, 0.66144515,
  0.66173948, 0.66206664, 0.66242619, 0.66281766, 0.66324058, 0.66369451,
  0.66417902, 0.66469367, 0.66523802, 0.66581167, 0.66641419, 0.66704517,
  0.66770422, 0.66839095, 0.66910495, 0.66984584, 0.67061326, 0.67140681,
  0.67222614, 0.67307088, 0.67394067, 0.67483517, 0.67575402, 0.67669687,
  0.67766339, 0.67865325, 0.67966612, 0.68070166, 0.68175955, 0.68283949,
  0.68394114, 0.68506421, 0.68620839, 0.68737336, 0.68855885, 0.68976453,
  0.69099013, 0.69223536, 0.69349992, 0.69478353, 0.69608592, 0.6974068,
  0.9907482,  0.98169891, 0.97284822, 0.96419228, 0.95572727, 0.94744946,
  0.93935517, 0.93144078, 0.92370273, 0.91613752, 0.90874171, 0.90151191,
  0.89444481, 0.88753715, 0.88078573, 0.8741874,  0.8677391,  0.86143779,
  0.85528051, 0.84926435, 0.84338646, 0.83764405, 0.83203438, 0.82655476,
  0.82120257, 0.81597521, 0.81087018, 0.805885,   0.80101724, 0.79626453,
  0.79162455, 0.78709503, 0.78267373, 0.77835847, 0.77414712, 0.77003759,
  0.76602783, 0.76211584, 0.75829965, 0.75457736, 0.75094707, 0.74740696,
  0.74395521, 0.74059008, 0.73730983, 0.73411278, 0.73099728, 0.72796172,
  0.7250045,  0.72212409, 0.71932195, 0.71660193, 0.7139626,  0.71140257,
  0.70892048, 0.70651499, 0.70418479, 0.70192861, 0.69974518, 0.69763329,
  0.69559172, 0.69361931, 0.69171489, 0.68987734, 0.68810556, 0.68639845,
  0.68475496, 0.68317405, 0.6816547,  0.68019591, 0.6787967,  0.67745612,
  0.67617322, 0.67494709, 0.67377682, 0.67266154, 0.67160036, 0.67059245,
  0.66963697, 0.6687331,  0.66788005, 0.66707703, 0.66632327, 0.66561801,
  0.66496052, 0.66435007, 0.66378595, 0.66326745, 0.6627939,  0.66236463,
  0.66197896, 0.66163626, 0.66133589, 0.66107724, 0.66085967, 0.66068261,
  0.66054545, 0.66044762, 0.66038855, 0.66036769, 0.66038449, 0.66043841,
  0.66052892, 0.66065551, 0.66081767, 0.6610149,  0.6612467,  0.6615126,
  0.66181212, 0.66214479, 0.66251017, 0.6629078,  0.66333724, 0.66379805,
  0.66428981, 0.66481211, 0.66536452, 0.66594665, 0.66655809, 0.66719846,
  0.66786737, 0.66856444, 0.66928929, 0.67004157, 0.67082092, 0.67162697,
  0.67245938, 0.67331781, 0.67420192, 0.67511137, 0.67604585, 0.67700502,
  0.67798857, 0.67899619, 0.68002758, 0.68108242, 0.68216042, 0.6832613,
  0.68438475, 0.68553049, 0.68669824, 0.68788774, 0.68909869, 0.69033084,
  0.69158392, 0.69285767, 0.69415183, 0.69546615, 0.69680037, 0.69815425,
  0.9907482,  0.98169891, 0.97284822, 0.96419228, 0.95572727, 0.94744946,
  0.93935517, 0.93144078, 0.92370273, 0.91613752, 0.90874171, 0.90151191,
  0.89444481, 0.88753715, 0.88078573, 0.8741874,  0.8677391,  0.86143779,
  0.85528051, 0.84926435, 0.84338646, 0.83764405, 0.83203438, 0.82655476,
  0.82120257, 0.81597521, 0.81087018, 0.805885,   0.80101724, 0.79626453,
  0.79162455, 0.78709503, 0.78267373, 0.77835847, 0.77414712, 0.77003759,
  0.76602783, 0.76211584, 0.75829965, 0.75457736, 0.75094707, 0.74740696,
  0.74395521, 0.74059008, 0.73730983, 0.73411278, 0.73099728, 0.72796172,
  0.7250045,  0.72212409, 0.71932195, 0.7165966,  0.71394661, 0.71137057,
  0.70886708, 0.70643479, 0.70407238, 0.70177856, 0.69955206, 0.69739162,
  0.69529605, 0.69326414, 0.69129474, 0.68938671, 0.68753892, 0.6857503,
  0.68401976, 0.68234626, 0.68072878, 0.67916632, 0.67765789, 0.67620253,
  0.67479929, 0.67344727, 0.67214554, 0.67089324, 0.66968949, 0.66853344,
  0.66742427, 0.66636116, 0.66534331, 0.66436994, 0.66344028, 0.66255359,
  0.66170913, 0.66090618, 0.66014404, 0.65942201, 0.65873941, 0.65809559,
  0.65748989, 0.65692167, 0.6563903,  0.65589518, 0.6554357,  0.65501128,
  0.65462132, 0.65426528, 0.65394258, 0.65365269, 0.65339506, 0.65316919,
  0.65297454, 0.65281061, 0.65267692, 0.65257297, 0.65249828, 0.65245239,
  0.65243484, 0.65244517, 0.65248294, 0.65254773, 0.65263909, 0.65275661,
  0.65289989, 0.6530685,  0.65326207, 0.65348019, 0.65372249, 0.65398859,
  0.65427811, 0.6545907,  0.65492599, 0.65528364, 0.6556633,  0.65606463,
  0.65648729, 0.65693097, 0.65739533, 0.65788006, 0.65838485, 0.65890938,
  0.65945336, 0.6600165,  0.66059849, 0.66119904, 0.66181789, 0.66245474,
  0.66310932, 0.66378136, 0.6644706,  0.66517677, 0.66589961, 0.66663887,
  0.66739429, 0.66816564, 0.66895265, 0.66975511, 0.67057276, 0.67140537],
  atol=1e-02)

    ####------- PLOTTING  -------------------------------------------------------------------------------------------------
    # Plot energy figure
#    plt.figure()
#    x = np.arange(0,1500,1)             # create x-axis length
#    plt.plot(x, energy[:1500], 'r')     # plot congruent condition
#    plt.plot(x, energy[1500:3000], 'b') # plot incongruent condition
#    plt.plot(x, energy[3000:4500], 'g') # plot neutral condition
#    plt.ylabel('ENERGY')                # add ylabel
#    plt.xlabel('cycles')                # add x label
#    legend = ['congruent', 'incongruent', 'neutral']
#    plt.legend(legend)
#    plt.show()


