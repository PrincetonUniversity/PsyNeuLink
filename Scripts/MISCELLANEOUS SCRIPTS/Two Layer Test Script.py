from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism import TransferMechanism

# from PsyNeuLink.Globals.Run import run, construct_inputs

Input_Layer = TransferMechanism(name='Input Layer',
                       function=Linear,
                       default_variable = np.zeros((2,)))

Output_Layer = TransferMechanism(name='Output Layer',
                        function=Logistic(),
                        default_variable = [0])

# Weights_matrix = lambda sender, receiver : random_matrix(sender, receiver, .2, -.1)
# Weights_matrix = (np.arange(1*2).reshape((1, 2)) + 1)/(2*5)
Weights_matrix = [[.1], [0.15]]
print(Weights_matrix)

# This projection will be used by the process below by referencing it in the process' pathway;
#    note: sender and receiver args don't need to be specified
Weights = MappingProjection(name='Weights',
                        matrix=Weights_matrix
                        )

my_process = process(default_variable=[0, 0],
                     # pathway=[Input_Layer, Weights, Output_Layer],
                     pathway=[Input_Layer, Output_Layer],
                     # clamp_input=SOFT_CLAMP,
                     learning=LEARNING,
                     learning_rate=1.0,
                     target=[1],
                     prefs={VERBOSE_PREF: False,
                            REPORT_OUTPUT_PREF: True})

stim_list = {Input_Layer:[[0, 0], [0, 1], [1, 0], [1, 1]]}
target_list = {Output_Layer:[[0], [1], [1], [0]]}


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# COMPOSITION = PROCESS
COMPOSITION = SYSTEM
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_target():

    if COMPOSITION is PROCESS:
        i = composition.input
        t = composition.target
    elif COMPOSITION is SYSTEM:
        i = composition.input
        t = composition.target_input_states[0].value
    print ('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(i, t))


if COMPOSITION is PROCESS:
    # z.execute()

    composition = my_process

    # PROCESS VERSION:
    my_process.run(num_trials=10,
                   # inputs=[[-1, 30],[2, 10]],
                   # targets=[[0, 0, 1],[0, 0, 1]],
                   inputs=stim_list,
                   targets=target_list,
                   call_before_trial=print_header,
                   call_after_trial=show_target)

elif COMPOSITION is SYSTEM:
    # SYSTEM VERSION:
    x = system(processes=[my_process],
               targets=[0],
               learning_rate=0.1)

    x.reportOutputPref = True
    composition = x

    # x.show_graph_with_learning()
    run_output = x.run(num_trials=10,
                       inputs=stim_list,
                       targets=target_list,
                       call_before_trial=print_header,
                       call_after_trial=show_target)

else:
    print ("Multilayer Learning Network NOT RUN")
