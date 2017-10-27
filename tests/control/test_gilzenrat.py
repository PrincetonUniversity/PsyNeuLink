"""
This implements a model of Locus Coeruleus / Norepinephrine (LC/NE) function described in `Gilzenrat et al. (2002)
<http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_, used to simulate behavioral
and electrophysiological data (from LC recordings) in non-human primates.

"""
import sys
import numpy as np

from psyneulink.library.subsystems.agt.gilzenrattransfermechanism import GilzenratTransferMechanism
from psyneulink.components.functions.function import Linear, Logistic, NormalDist, FHNIntegrator
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.library.subsystems.agt.lccontrolmechanism import LCControlMechanism

class TestGilzenratMechanisms:

    def test_defaults(self):
        G = GilzenratTransferMechanism()

        # - - - - - LCA integrator functions - - - - -
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
        G = GilzenratTransferMechanism(
                            initial_value=np.array([[1.0]]),
                            function = Linear(slope=2.0)
                            # time_step_size = default = 0.02
                            # noise = default = 0.0
                            )
        P = Process(pathway=[G])
        S = System(processes=[P])
        G.output_state.value = [0.0]

        # - - - - - LCA integrator functions - - - - -
        # X = previous_value + (rate * previous_value + variable) * self.time_step_size + noise
        # f(X) = 2.0*X + 0

        # - - - - - starting values - - - - -
        # variable = G.output_state.value + stimulus = 0.0 + 1.0 = 1.0
        # previous_value = initial_value = 1.0

        np.testing.assert_allclose(S.execute([[1.0]]), np.array([[2.0]]))
        # X = 1.0 + (-1.0 + 1.0)*0.02 + 0.0
        # X = 1.0 + 0.0 + 0.0 = 1.0 <--- previous value 1.0
        # f(X) = 2.0*1.0  <--- return 2.0, recurrent projection 2.0

        np.testing.assert_allclose(S.execute([[1.0]]), np.array([[2.08]]))
        # X = 1.0 + (-1.0 + 3.0)*0.02 + 0.0
        # X = 1.0 + 0.04 = 1.04 <--- previous value 1.04
        # f(X) = 2.0*1.04  <--- return 2.08

        np.testing.assert_allclose(S.execute([[1.0]]), np.array([[2.1616]]))
        # X = 1.04 + (-1.04 + 3.08)*0.02 + 0.0
        # X = 1.04 + 0.0408 = 1.0808 <--- previous value 1.0808
        # f(X) = 2.1616  <--- return 2.1616

    def test_fhn_gilzenrat_figure_2(self):
        # Isolate the FHN mechanism for testing and recreate figure 2 from the gilzenrat paper

        initial_v = 0.2
        initial_w = 0.0

        F = IntegratorMechanism(
            name='IntegratorMech-FHNFunction',
            function=FHNIntegrator(
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

        np.testing.assert_allclose(plot_v_list, [0.2, 0.22493312915681499, 0.24844236992931412, 0.27113468959297515,
                                                 0.29350254152625221, 0.31599112332052792, 0.33904651470437225,
                                                 0.36315614063656521, 0.38888742632665502, 0.41692645840176923,
                                                 0.44811281741549686]
)
        np.testing.assert_allclose(plot_w_list, [0.0, 0.0019518690642000148, 0.0041351416812363193,
                                                 0.0065323063637677276, 0.0091322677555586273, 0.011929028036111457,
                                                 0.014921084302726394, 0.018111324713170868, 0.021507331976846619,
                                                 0.025122069034563425, 0.028974949616469712]
)
