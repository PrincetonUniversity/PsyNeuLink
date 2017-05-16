from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection, TARGET_MSE
from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.TimeScale import TimeScale

def test_multilayer():
    Input_Layer = TransferMechanism(
        name='Input Layer',
        function=Logistic,
        default_input_value = np.zeros((2,)),
    )

    Hidden_Layer_1 = TransferMechanism(
        name='Hidden Layer_1',
        function=Logistic(),
        default_input_value = np.zeros((5,)),
    )

    Hidden_Layer_2 = TransferMechanism(
        name='Hidden Layer_2',
        function=Logistic(),
        default_input_value = [0,0,0,0],
    )

    Output_Layer = TransferMechanism(
        name='Output Layer',
        function=Logistic,
        default_input_value = [0,0,0],
    )

    Input_Weights_matrix = (np.arange(2*5).reshape((2, 5)) + 1)/(2*5)
    Middle_Weights_matrix = (np.arange(5*4).reshape((5, 4)) + 1)/(5*4)
    Output_Weights_matrix = (np.arange(4*3).reshape((4, 3)) + 1)/(4*3)


    # TEST PROCESS.LEARNING WITH:
    # CREATION OF FREE STANDING PROJECTIONS THAT HAVE NO LEARNING (Input_Weights, Middle_Weights and Output_Weights)
    # INLINE CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)
    # NO EXPLICIT CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)

    # This projection will be used by the process below by referencing it in the process' pathway;
    #    note: sender and receiver args don't need to be specified
    Input_Weights = MappingProjection(
        name='Input Weights',
        matrix=Input_Weights_matrix,
    )

    # This projection will be used by the process below by assigning its sender and receiver args
    #    to mechanismss in the pathway
    Middle_Weights = MappingProjection(
        name='Middle Weights',
        sender=Hidden_Layer_1,
        receiver=Hidden_Layer_2,
        matrix=Middle_Weights_matrix,
    )

    # Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
    Output_Weights = MappingProjection(
        name='Output Weights',
        sender=Hidden_Layer_2,
        receiver=Output_Layer,
        matrix=Output_Weights_matrix,
    )

    p = process(
        default_input_value=[0, 0],
        pathway=[
            Input_Layer,
            # The following reference to Input_Weights is needed to use it in the pathway
            #    since it's sender and receiver args are not specified in its declaration above
            Input_Weights,
            Hidden_Layer_1,
            # No projection specification is needed here since the sender arg for Middle_Weights
            #    is Hidden_Layer_1 and its receiver arg is Hidden_Layer_2
            # Middle_Weights,
            Hidden_Layer_2,
            # Output_Weights does not need to be listed for the same reason as Middle_Weights
            # If Middle_Weights and/or Output_Weights is not declared above, then the process
            #    will assign a default for missing projection
            # Output_Weights,
            Output_Layer
        ],
        clamp_input=SOFT_CLAMP,
        learning=LEARNING,
        learning_rate=1.0,
        target=[0,0,1],
        prefs={
            VERBOSE_PREF: False,
            REPORT_OUTPUT_PREF: False
        },
    )

    stim_list = {Input_Layer:[[-1, 30]]}
    target_list = {Output_Layer:[[0, 0, 1]]}


    def print_header():
        print("\n\n**** TRIAL: ", CentralClock.trial)

    def show_target():
        i = s.input
        t = s.targetInputStates[0].value
        print ('\nOLD WEIGHTS: \n')
        print ('- Input Weights: \n', Input_Weights.matrix)
        print ('- Middle Weights: \n', Middle_Weights.matrix)
        print ('- Output Weights: \n', Output_Weights.matrix)
        print ('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(i, t))
        print ('ACTIVITY FROM OLD WEIGHTS: \n')
        print ('- Middle 1: \n', Hidden_Layer_1.value)
        print ('- Middle 2: \n', Hidden_Layer_2.value)
        print ('- Output:\n', Output_Layer.value)

    s = system(
        processes=[p],
        targets=[0, 0, 1],
        learning_rate=1.0,
    )

    s.reportOutputPref = True

    results = s.run(
        num_executions=10,
        inputs=stim_list,
        targets=target_list,
        call_before_trial=print_header,
        call_after_trial=show_target,
    )

    objective_output_layer = s.mechanisms[4]

    results_list = []
    for elem in s.results:
        for nested_elem in elem:
            nested_elem = nested_elem.tolist()
            try:
                iter(nested_elem)
            except TypeError:
                nested_elem = [nested_elem]
            results_list.extend(nested_elem)

    expected_results_list = [
        0.83448370231346, 0.8707201786695044, 0.8999743264756163,
        0.8683927358195268,
        0.0007175454706347559,
        0.7797019346587708, 0.832631377973685, 0.9015962725334999,
        0.8379765283886519,
        0.0024906569174657483,
        0.7021850243581587, 0.7773822989749413, 0.9030776464400238,
        0.7942149899243746,
        0.0068679773434540785,
        0.60279149220922, 0.6995807898859298, 0.9045314296553872,
        0.7356345705835124,
        0.01582443585963084,
        0.49679270296011985, 0.6003032086020696, 0.9061008242664779,
        0.6677322452762224,
        0.03019552718795884,
        0.40562020017560557, 0.49472390987426346, 0.9078661674551524,
        0.6027367591683405,
        0.04787522308107376,
        0.3376302465351187, 0.4039763747743685, 0.9097767517554378,
        0.550461124354975,
        0.06528749483581726,
        0.28892812402413326, 0.33633532241586245, 0.9117192979826672,
        0.512327581474221,
        0.08013144535100698,
        0.25348770666261206, 0.287918960919678, 0.9136124968123867,
        0.4850063881315589,
        0.09204918341087988,
        0.22686074091309802, 0.25270211784499447, 0.9154214931287058,
        0.4649947839622661,
        0.10155340629221028
    ]

    expected_output = [
        (Output_Layer.output_states.values, [np.array([ 0.22686074,  0.25270212,  0.91542149])]),
        (objective_output_layer.output_states[TARGET_MSE].value, np.array(0.04082589331852094)),
        (Input_Weights.matrix, np.array([
            [0.09890269, 0.19810968, 0.29740194,  0.39678767, 0.49627111],
            [0.5959199,  0.69297125, 0.79033968, 0.88805564, 0.98613492],
        ])),
        (Middle_Weights.matrix, np.array([
            [ 0.08992499,  0.10150104,  0.11891032,  0.14250149],
            [ 0.29158517,  0.30154765,  0.31758943,  0.34007336],
            [ 0.49318268,  0.50159531,  0.51632339,  0.5377435 ],
            [ 0.69471052,  0.70164382,  0.71511777,  0.73552215],
            [ 0.8961628 ,  0.90169281,  0.91397691,  0.93341744]
       ])),
        (Output_Weights.matrix, np.array([
            [-0.71039394, -0.66929423,  0.31014399],
            [-0.47462798, -0.43340256,  0.56113343],
            [-0.2388705 , -0.19778374,  0.81214434],
            [-0.00287122,  0.03785105,  1.06315816]
        ])),
        (results_list, expected_results_list)
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
