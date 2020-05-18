import numpy as np

from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import FitzHughNagumoIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.compositions.composition import Composition
from psyneulink.library.components.mechanisms.processing.transfer.lcamechanism import LCAMechanism


class TestGilzenratMechanisms:

    def test_defaults(self):
        G = LCAMechanism(integrator_mode=True,
                         leak=1.0,
                         noise=0.0,
                         time_step_size=0.02,
                         function=Linear,
                         self_excitation=1.0,
                         competition=-1.0)

        # - - - - - LCAMechanism integrator functions - - - - -
        # X = previous_value + (rate * previous_value + variable) * self.time_step_size + noise
        # f(X) = 1.0*X + 0

        np.testing.assert_allclose(G.execute(), np.array([[0.0]]))
        # X = 0.0 + (0.0 + 0.0)*0.02 + 0.0
        # X = 0.0 <--- previous value 0.0
        # f(X) = 1.0*0.0  <--- return 0.0, recurrent projection 0.0

        np.testing.assert_allclose(G.execute(1.0), np.array([[0.02]]))
        # X = 0.0 + (0.0 + 1.0)*0.02 + 0.0
        # X = 0.02 <--- previous value 0.02
        # f(X) = 1.0*0.02  <--- return 0.02, recurrent projection 0.02

        # Outside of a system, previous value works (integrator) but recurrent projection does NOT
        np.testing.assert_allclose(G.execute(1.0), np.array([[0.0396]]))
        # X = 0.02 + (-0.02 + 1.0)*0.02 + 0.0
        # X = 0.0396 --- previous value 0.0396
        # f(X) = 1.0*0.0396 <--- return 0.02, recurrent projection 0.02

    def test_previous_value_stored(self):
        G = LCAMechanism(integrator_mode=True,
                         leak=1.0,
                         noise=0.0,
                         time_step_size=0.02,
                         function=Linear(slope=2.0),
                         self_excitation=1.0,
                         competition=-1.0,
                         initial_value=np.array([[1.0]]))

        C = Composition(pathways=[G])
        G.output_port.value = [0.0]

        # - - - - - LCAMechanism integrator functions - - - - -
        # X = previous_value + (rate * previous_value + variable) * self.time_step_size + noise
        # f(X) = 2.0*X + 0

        # - - - - - starting values - - - - -
        # variable = G.output_port.value + stimulus = 0.0 + 1.0 = 1.0
        # previous_value = initial_value = 1.0
        # single_run = S.execute([[1.0]])
        # np.testing.assert_allclose(single_run, np.array([[2.0]]))
        np.testing.assert_allclose(C.execute(inputs={G:[[1.0]]}), np.array([[2.0]]))
        # X = 1.0 + (-1.0 + 1.0)*0.02 + 0.0
        # X = 1.0 + 0.0 + 0.0 = 1.0 <--- previous value 1.0
        # f(X) = 2.0*1.0  <--- return 2.0, recurrent projection 2.0

        np.testing.assert_allclose(C.execute(inputs={G:[[1.0]]}), np.array([[2.08]]))
        # X = 1.0 + (-1.0 + 3.0)*0.02 + 0.0
        # X = 1.0 + 0.04 = 1.04 <--- previous value 1.04
        # f(X) = 2.0*1.04  <--- return 2.08

        np.testing.assert_allclose(C.execute(inputs={G:[[1.0]]}), np.array([[2.1616]]))
        # X = 1.04 + (-1.04 + 3.08)*0.02 + 0.0
        # X = 1.04 + 0.0408 = 1.0808 <--- previous value 1.0808
        # f(X) = 2.1616  <--- return 2.1616

    def test_fitzHughNagumo_gilzenrat_figure_2(self):
        # Isolate the FitzHughNagumo mechanism for testing and recreate figure 2 from the gilzenrat paper

        initial_v = 0.2
        initial_w = 0.0

        F = IntegratorMechanism(
            name='IntegratorMech-FitzHughNagumoFunction',
            function=FitzHughNagumoIntegrator(
                initial_v=initial_v,
                initial_w=initial_w,
                time_step_size=0.01,
                time_constant_w=1.0,
                time_constant_v=0.01,
                a_v=-1.0,
                b_v=1.0,
                c_v=1.0,
                d_v=0.0,
                e_v=-1.0,
                f_v=1.0,
                threshold=0.5,
                mode=1.0,
                uncorrelated_activity=0.0,
                a_w=1.0,
                b_w=-1.0,
                c_w=0.0

            )
        )
        plot_v_list = [initial_v]
        plot_w_list = [initial_w]

        # found this stimulus by guess and check b/c one was not provided with Figure 2 params
        stimulus = 0.073
        # increase range to 200 to match Figure 2 in Gilzenrat
        for i in range(10):
            results = F.execute(stimulus)
            plot_v_list.append(results[0][0][0])
            plot_w_list.append(results[1][0][0])

        # ** uncomment the lines below if you want to view the plot:
        # from matplotlib import pyplot as plt
        # plt.plot(plot_v_list)
        # plt.plot(plot_w_list)
        # plt.show()

        np.testing.assert_allclose(plot_v_list, [0.2, 0.22493312915681499, 0.24840327807265583, 0.27101619694032797,
                                                 0.29325863380332173, 0.31556552465130933, 0.33836727470568129,
                                                 0.36212868305470697, 0.38738542852040492, 0.41478016676749552,
                                                 0.44509530539552955]
)
        print(plot_w_list)
        np.testing.assert_allclose(plot_w_list, [0.0, 0.0019900332500000003, 0.0042083541185625045,
                                                 0.0066381342093118408, 0.009268739886338381, 0.012094486544132229,
                                                 0.015114073825358726, 0.018330496914962583, 0.021751346023501487,
                                                 0.025389465931011893, 0.029263968140538919]

)
#
# class TestGilzenratFullModel:
#     def test_replicate_gilzenrat_paper(self):
#         """
#         This implements a model of Locus Coeruleus / Norepinephrine (LC/NE) function described in `Gilzenrat et al. (2002)
#         <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_, used to simulate behavioral
#         and electrophysiological data (from LC recordings) in non-human primates.
#
#         This test does NOT validate output against expected values yet -- it is only here to ensure that changes to
#         PsyNeuLink do not prevent the system below from being created without errors
#
#         Plotting code is commented out and only one trial will be executed
#         """
#         # --------------------------------- Global Variables ----------------------------------------
#
#         # Mode ("coherence")
#         C = 0.95
#         # Uncorrelated Activity
#         d = 0.5
#
#         # Initial values
#         initial_h_of_v = 0.07
#         # initial_h_of_v = 0.07
#         initial_v = (initial_h_of_v - (1 - C) * d) / C
#         # initial_w = 0.14
#         initial_w = 0.14
#
#         # g(t) = G + k*w(t)
#
#         # Scaling factor for transforming NE release (u ) to gain (g ) on potentiated units
#         k = 3.0
#         # Base level of gain applied to decision and response units
#         G = 0.5
#
#         # numerical integration
#         time_step_size = 0.02
#         # number_of_trials = int(20/time_step_size)
#         number_of_trials = 1
#
#         # noise
#         standard_deviation = 0.22 * (time_step_size ** 0.5)
#
#         # --------------------------------------------------------------------------------------------
#
#         input_layer = TransferMechanism(default_variable=np.array([[0, 0]]),
#                                         name='INPUT LAYER')
#
#         # Implement projections from inputs to decision layer with weak cross-talk connections
#         #    from target and distractor inputs to their competing decision layer units
#         input_weights = np.array([[1, .33], [.33, 1]])
#
#         # Implement self-excitatory (auto) and mutually inhibitory (hetero) connections within the decision layer
#         decision_layer = GilzenratTransferMechanism(size=2,
#                                                     initial_value=np.array([[1, 0]]),
#                                                     matrix=np.matrix([[1, 0], [0, -1]]),
#                                                     # auto=1.0,
#                                                     # hetero=-1.0,
#                                                     time_step_size=time_step_size,
#                                                     noise=NormalDist(mean=0.0,
#                                                                      standard_deviation=standard_deviation).function,
#                                                     function=Logistic(bias=0.0),
#                                                     name='DECISION LAYER')
#
#         # Implement connection from target but not distractor unit in decision layer to response
#         output_weights = np.array([[1.84], [0]])
#
#         # Implement response layer with a single, self-excitatory connection
#         # To do Markus: specify recurrent self-connrection weight for response unit to 2.00
#         response = GilzenratTransferMechanism(size=1,
#                                               initial_value=np.array([[2.0]]),
#                                               matrix=np.matrix([[0.5]]),
#                                               function=Logistic(bias=2),
#                                               time_step_size=time_step_size,
#                                               noise=NormalDist(mean=0.0, standard_deviation=standard_deviation).function,
#                                               name='RESPONSE')
#
#         # Implement response layer with input_port for ObjectiveMechanism that has a single value
#         # and a MappingProjection to it that zeros the contribution of the decision unit in the decision layer
#         LC = LCControlMechanism(
#             time_step_size_FitzHughNagumo=time_step_size,  # integrating step size
#             mode_FitzHughNagumo=C,  # coherence: set to either .95 or .55
#             uncorrelated_activity_FitzHughNagumo=d,  # Baseline level of intrinsic, uncorrelated LC activity
#             time_constant_v_FitzHughNagumo=0.05,
#             time_constant_w_FitzHughNagumo=5,
#             a_v_FitzHughNagumo=-1.0,
#             b_v_FitzHughNagumo=1.0,
#             c_v_FitzHughNagumo=1.0,
#             d_v_FitzHughNagumo=0.0,
#             e_v_FitzHughNagumo=-1.0,
#             f_v_FitzHughNagumo=1.0,
#             a_w_FitzHughNagumo=1.0,
#             b_w_FitzHughNagumo=-1.0,
#             c_w_FitzHughNagumo=0.0,
#             t_0_FitzHughNagumo=0,
#             initial_v_FitzHughNagumo=initial_v,
#             initial_w_FitzHughNagumo=initial_w,
#             threshold_FitzHughNagumo=0.5,
#             # Parameter describing shape of the FitzHugh–Nagumo cubic nullcline for the fast excitation variable v
#             objective_mechanism=ObjectiveMechanism(
#                 function=Linear,
#                 monitor=[(decision_layer, None, None, np.array([[0.3], [0.0]]))],
#                 # monitor=[{PROJECTION_TYPE: MappingProjection,
#                 #                           SENDER: decision_layer,
#                 #                           MATRIX: np.array([[0.3],[0.0]])}],
#                 name='LC ObjectiveMechanism'
#             ),
#             modulated_mechanisms=[decision_layer, response],
#             name='LC')
#
#         for signal in LC._control_signals:
#             signal._intensity = k * initial_w + G
#
#         # ELICITS WARNING:
#         decision_process = Process(pathway=[input_layer,
#                                             input_weights,
#                                             decision_layer,
#                                             output_weights,
#                                             response],
#                                    name='DECISION PROCESS')
#
#         lc_process = Process(pathway=[decision_layer,
#                                       # CAUSES ERROR:
#                                       # np.array([[1,0],[0,0]]),
#                                       LC],
#                              name='LC PROCESS')
#
#         task = System(processes=[decision_process, lc_process])
#
#         # stimulus
#         stim_list_dict = {input_layer: np.repeat(np.array([[0, 0], [1, 0]]), 10 / time_step_size, axis=0)}
#
#         def h_v(v, C, d):
#             return C * v + (1 - C) * d
#
#         # Initialize output arrays for plotting
#         LC_results_v = [h_v(initial_v, C, d)]
#         LC_results_w = [initial_w]
#         decision_layer_target = [0.5]
#         decision_layer_distractor = [0.5]
#         response_layer = [0.5]
#
#         def record_trial():
#             LC_results_v.append(h_v(LC.value[2][0], C, d))
#             LC_results_w.append(LC.value[3][0])
#             decision_layer_target.append(decision_layer.value[0][0])
#             decision_layer_distractor.append(decision_layer.value[0][1])
#             response_layer.append(response.value[0][0])
#             current_trial_num = len(LC_results_v)
#             if current_trial_num % 50 == 0:
#                 percent = int(round((float(current_trial_num) / number_of_trials) * 100))
#                 sys.stdout.write("\r" + str(percent) + "% complete")
#                 sys.stdout.flush()
#
#         sys.stdout.write("\r0% complete")
#         sys.stdout.flush()
#         task.run(stim_list_dict, num_trials=number_of_trials, call_after_trial=record_trial)
#
#         # from matplotlib import pyplot as plt
#         # import numpy as np
#         # t = np.arange(0.0, len(LC_results_v), 1.0)
#         # plt.plot(t, LC_results_v, label="h(v)")
#         # plt.plot(t, LC_results_w, label="w")
#         # plt.plot(t, decision_layer_target, label="target")
#         # plt.plot(t, decision_layer_distractor, label="distractor")
#         # plt.plot(t, response_layer, label="response")
#         # plt.xlabel(' # of timesteps ')
#         # plt.ylabel('h(V)')
#         # plt.legend(loc='upper left')
#         # plt.ylim((-0.2, 1.2))
#         # plt.show()
#
#         # This prints information about the System,
#         # including its execution list indicating the order in which the Mechanisms will execute
#         # IMPLEMENTATION NOTE:
#         #  MAY STILL NEED TO SCHEDULE RESPONSE TO EXECUTE BEFORE LC
#         #  (TO BE MODULATED BY THE GAIN MANIPULATION IN SYNCH WITH THE DECISION LAYER
#         # task.show()
#
#         # This displays a diagram of the System
#         # task.show_graph()
#
