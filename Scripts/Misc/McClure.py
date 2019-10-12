import functools
import numpy as np
import psyneulink as pnl

import psyneulink.core.components.functions.transferfunctions

np.random.seed(2)

a = 0.50        # Parameter describing shape of the FitzHugh–Nagumo cubic nullcline for the fast excitation variable v
d = 0.50        # Baseline level of intrinsic, uncorrelated LC activity
G = 0.50        # Base level of gain applied to decision and response units
k = 3.00        # Scaling factor for transforming NE release (u) to gain (g) on potentiated units

#SD = 0.1        # Standard deviation of Gaussian noise distributions | NOTE: 0.22 in Gilzenrat paper

tau_v = 0.05    # Time constant for fast LC excitation variable v | NOTE: tau_v is misstated in the Gilzenrat paper(0.5)
tau_u = 5.00    # Time constant for slow LC recovery variable (‘NE release’) u
dt = 0.02       # Time step size for numerical integration

C = 0.95           # Mode ("coherence") - high
initial_hv = 0.07  # Initial value for h(v)
initial_u = 0.14   # initial value u

# C = 0.55           # Mode ("coherence") - low
# initial_hv = 0.2   # Initial value for h(v) with low C
# initial_u = 0.2    # Initial value for u with low C

initial_v = (initial_hv - (1 - C) * d) / C  # get initial v from initial h(v)


input_layer = pnl.TransferMechanism(size=2,
                                    name='INPUT LAYER')


action_selection = pnl.TransferMechanism(size=2,
                                         function=psyneulink.core.components.functions.transferfunctions.SoftMax(
                                           output=pnl.ALL,
                                           gain=1.0),
                                         output_ports=[{pnl.NAME: 'SELECTED ACTION',
                                                         pnl.VARIABLE: [(pnl.INPUT_PORT_VARIABLES, 0),
                                                                        (pnl.OWNER_VALUE, 0)],
                                                         # pnl.VARIABLE: [(pnl.OWNER_VALUE, 0)],
                                                         pnl.FUNCTION: psyneulink.core.components.functions
                                         .selectionfunctions.OneHot(mode=pnl.PROB_INDICATOR).function},
                                                        {pnl.NAME: 'REWARD RATE',
                                                         # pnl.VARIABLE: [pnl.OWNER_VALUE],
                                                         pnl.VARIABLE: [(pnl.OWNER_VALUE,0)],
                                                         pnl.FUNCTION: psyneulink.core.components.functions
                                         .integratorfunctions.AdaptiveIntegrator(rate=0.2)},
                                                        {pnl.NAME: 'CONFLICT K',
                                                         # pnl.VARIABLE: [pnl.OWNER_VALUE],
                                                         pnl.VARIABLE: [(pnl.OWNER_VALUE,0)],
                                          #Jon said this should also work and would be safer: [(pnl.OWNER_VALUE, 0),
                                          #(pnl.OWNER_VALUE, 1)], but it doesn't work (maybe I did sth wrong)
                                                         pnl.FUNCTION: psyneulink.core.components.functions
                                         .objectivefunctions.Stability(default_variable=[0, 0],
                                                                                                                                         metric=pnl.ENERGY,
                                                                                                                                         normalize=True)},
                                                        ],
                                                            #as stated in the paper 'Response conflict was calculated as a normalized                                                                   measure of the energy in the response units during the trial'
                                         name='Action Selection')





# rate = pnl.ObjectiveMechanism(monitored_output_ports=[action_selection.output_ports[0]],
#                               function=pnl.AdaptiveIntegrator(rate=0.2,
#                                                               noise=reward,
#                                                               time_step_size=0.02),
#                               name='REWARD RATE')

# K = pnl.ObjectiveMechanism(#size=1,
#                           monitored_output_ports=[action_selection.output_port],
#                           function=pnl.Stability(metric=pnl.ENERGY,
#                                                  normalize=True),
#                           name='K')

conflicts = pnl.IntegratorMechanism(input_ports=[action_selection.output_ports[2]],
                                    function=psyneulink.core.components.functions.statefulfunctions.integratorfunctions.DualAdaptiveIntegrator(short_term_gain=6.0,
                                                                                                                                                long_term_gain=6.0,
                                                                                                                                                short_term_rate=0.05,
                                                                                                                                                long_term_rate=0.2),
                                    name='Short- and Long-term conflict')

decision_process = pnl.Process(default_variable=[0, 0],
                               pathway=[input_layer,
                                        action_selection],
                               learning=pnl.LearningProjection(learning_function=psyneulink.core.components.functions
                                   .learningfunctions.Reinforcement(
                                   learning_rate=0.03)), # if learning rate set to .3 OutputPort values annealing to [0., 0.]
                               # which leads to error in reward function
                               target=0
                               )

print('reward prediction weights: \n', action_selection.input_port.path_afferents[0].matrix)
print('target_mechanism weights: \n', action_selection.output_port.efferents[0].matrix)

conflict_process = pnl.Process(pathway=[action_selection, conflicts])

LC_NE = pnl.LCControlMechanism(objective_mechanism=pnl.ObjectiveMechanism(monitored_output_ports=[action_selection],
                                                                          name='LC-NE ObjectiveMech'),
                               modulated_mechanisms=[action_selection],
                               integration_method='EULER',
                               initial_w_FitzHughNagumo=initial_u,
                               initial_v_FitzHughNagumo=initial_v,
                               time_step_size_FitzHughNagumo=dt,
                               t_0_FitzHughNagumo=0.0,
                               a_v_FitzHughNagumo=-1.0,
                               b_v_FitzHughNagumo=1.0,
                               c_v_FitzHughNagumo=1.0,
                               d_v_FitzHughNagumo=0.0,
                               e_v_FitzHughNagumo=-1.0,
                               f_v_FitzHughNagumo=1.0,
                               time_constant_v_FitzHughNagumo=tau_v,
                               a_w_FitzHughNagumo=1.0,
                               b_w_FitzHughNagumo=-1.0,
                               c_w_FitzHughNagumo=0.0,
                               threshold_FitzHughNagumo=a,
                               time_constant_w_FitzHughNagumo=tau_u,
                               mode_FitzHughNagumo=C,
                               uncorrelated_activity_FitzHughNagumo=d,
                               base_level_gain=G,
                               scaling_factor_gain=k,
                               name='LC-NE')

updateC = pnl.ControlMechanism(objective_mechanism=pnl.ObjectiveMechanism(
    monitor_for_control=[action_selection.output_ports[1], conflicts.output_port]),
    control_signals=[LC_NE.parameter_ports[35]],
    name='C Update')

update_process = pnl.Process(pathway=[LC_NE],
                             name='UPDATE PROCESS')


actions = ['left', 'right']
reward_values = np.array([1, 0])
first_reward = 0

action_selection.output_port.value = [0, 1]


def reward():
    print(reward_values, action_selection.output_port.value)
    return [reward_values[int(np.nonzero(action_selection.output_port.value)[0])]]


rrate = [0.]
conflictK = [0.]
coherence = [0.]
update = [0.]
cons = [0.]


def print_header(system):
    print("\n\n**** Time: ", system.scheduler.clock.simple_time)


def show_weights(system):
    # print('Reward prediction weights: \n', action_selection.input_port.path_afferents[0].matrix)
    # print(
    #     '\nAction selected:  {}; predicted reward: {}'.format(
    #         np.nonzero(action_selection.output_port.value)[0][0],
    #         action_selection.output_port.value[np.nonzero(action_selection.output_port.value)][0]
    #     )
    assert True
    comparator = action_selection.output_port.efferents[0].receiver.owner
    learn_mech = action_selection.output_port.efferents[1].receiver.owner
    print('\n'
          '\naction_selection value:     {} '
          '\naction_selection output:    {} '
          '\ncomparator sample:          {} '
          '\ncomparator target:          {} '
          '\nlearning mech act in:       {} '
          '\nlearning mech act out:      {} '
          '\nlearning mech error in:     {} '
          '\nlearning mech error out:    {} '
          '\nlearning mech learning_sig: {} '
          '\npredicted reward:           {} '
          '\nreward rate:                {} '
          '\nconflict K:                 {} '
          '\ncoherence C:                {} '
          '\nshort-long-term conflict:   {} '.
        format(
            action_selection.parameters.value.get(system),
            action_selection.output_port.parameters.value.get(system),
            comparator.input_ports[pnl.SAMPLE].parameters.value.get(system),
            comparator.input_ports[pnl.TARGET].parameters.value.get(system),
            learn_mech.input_ports[pnl.ACTIVATION_INPUT].parameters.value.get(system),
            learn_mech.input_ports[pnl.ACTIVATION_OUTPUT].parameters.value.get(system),
            learn_mech.input_ports[pnl.ERROR_SIGNAL].parameters.value.get(system),
            learn_mech.output_ports[pnl.ERROR_SIGNAL].parameters.value.get(system),
            learn_mech.output_ports[pnl.LEARNING_SIGNAL].parameters.value.get(system),
            action_selection.output_port.parameters.value.get(system)[np.nonzero(action_selection.output_port.parameters.value.get(system))][0],
            rrate.append(action_selection.output_ports[1].parameters.value.get(system)),
            conflictK.append(action_selection.output_ports[2].parameters.value.get(system)),
            coherence.append(LC_NE.parameter_ports[35].parameters.value.get(system)),
            update.append(updateC.output_port.parameters.value.get(system)),
            cons.append(conflicts.output_port.parameters.value.get(system))
                )
    )


decision_process.run(num_trials=10,
                     inputs=[[1, 1]],
                     targets=reward
                     )


# inputs = np.tile(np.repeat(np.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]]), 20, axis=0), (4, 1))
# input_dict = {input_layer: inputs}
input_dict = {input_layer: [1, 0]}

DA_sys = pnl.System(
    processes=[decision_process, conflict_process,
               update_process],
    controller=updateC,
    targets=[0],
    name='NE-DA System'
)

DA_sys.show_graph(show_learning=pnl.ALL,
                  show_control=pnl.ALL,
                  show_dimensions=True,
                  show_mechanism_structure=pnl.ALL)

DA_sys.run(
    num_trials=10,
    inputs=input_dict,
    targets=reward,
    call_after_trial=functools.partial(show_weights, DA_sys)
)

