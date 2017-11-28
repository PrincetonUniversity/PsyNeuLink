import numpy as np
import pytest
import random
import copy
try:
    import leabra
    leabra_available = True
except ImportError:
    leabra_available = False

from psyneulink.library.mechanisms.processing.leabramechanism import LeabraMechanism, build_leabra_network, train_leabra_network
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.functions.function import Linear, Logistic
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.globals.keywords import LEARNING

@pytest.mark.skipif(not leabra_available, reason='Leabra package is unavailable')
class TestLeabraMechanismInit:

    def test_leabra_init_empty(self):
        L = LeabraMechanism()
        val = L.execute([[10], [0]])
        assert len(val) == 1 and len(val[0]) == 1
        assert "LeabraMechanism" in L.name
        assert L.input_size == 1
        assert L.output_size == 1
        assert val[0][0] > 0.9

    # this kind of test (execute when input_size != output_size) does not work while np.atleast_2d is being used
    # in mechanism.py. When this is fixed, I should return and reinstate these tests 11/3/17 CW
    # def test_leabra_init_input(self):
    #     L = LeabraMechanism(input_size=5)
    #     assert L.input_size == 5
    #     assert L.output_size == 1
    #     val = L.execute([[1, 2, 3, 4, 5], [0]])
    #     assert val.tolist() == [0]

    def test_leabra_init_input_output(self):
        L = LeabraMechanism(input_size=3, output_size=3, name = 'L1')
        val = L.execute([[0, 0, 0], [0, 0, 0]])
        assert L.input_size == 3
        assert L.output_size == 3
        assert L.name == 'L1'
        assert val.tolist() == [[0, 0, 0]]

    # NOTE 11/3/17 CW: I have no intuition about what these values should be, so I'm not "testing" output values for now
    def test_leabra_init_no_hidden_sizes(self):
        L = LeabraMechanism(input_size=4, output_size=4, hidden_layers=2, training_flag=False)
        val = L.execute([[1, 2, 3, 4], [0, 0, 0, 0]])
        assert L.hidden_layers == 2
        assert L.hidden_sizes[0] == 4
        assert len(val[0]) == 4

# identical to test_leabra_network; but run_network has a different name to avoid pytest collisions
@pytest.mark.skipif(not leabra_available, reason='Leabra package is unavailable')
def run_network(network, input_pattern):
    assert len(network.layers[0].units) == len(input_pattern)
    network.set_inputs({'input_layer': input_pattern})

    network.trial()
    return [unit.act_m for unit in network.layers[-1].units]

@pytest.mark.skipif(not leabra_available, reason='Leabra package is unavailable')
class TestLeabraMechanismPrecision:

    def test_leabra_prec_no_train(self):
        in_size = 4
        out_size = 4
        num_hidden = 1
        num_trials = 2
        train = False
        inputs = [[0, 1, -1, 2]] * num_trials
        train_data = [[0] * out_size] * num_trials
        precision = 0.00001  # how far we accept error between PNL and Leabra output
        random_seed = 1  # because Leabra network initializes with small random weights
        random.seed(random_seed)
        L_spec = LeabraMechanism(input_size=in_size, output_size=out_size, hidden_layers=num_hidden, training_flag=train)
        random.seed(random_seed)
        leabra_net = build_leabra_network(in_size, out_size, num_hidden, None, train)
        leabra_net2 = copy.deepcopy(leabra_net)
        L_net = LeabraMechanism(leabra_net2)
        # leabra_net should be identical to the network inside L_net

        T1_spec = TransferMechanism(name='T1', size=in_size, function=Linear)
        T2_spec = TransferMechanism(name='T2', size=out_size, function=Linear)
        T1_net = TransferMechanism(name='T1', size=in_size, function=Linear)
        T2_net = TransferMechanism(name='T2', size=out_size, function=Linear)

        p1_spec = Process(pathway=[T1_spec, L_spec])
        proj_spec = MappingProjection(sender=T2_spec, receiver=L_spec.input_states[1])
        p2_spec = Process(pathway=[T2_spec, proj_spec, L_spec])
        s_spec = System(processes=[p1_spec, p2_spec])

        p1_net = Process(pathway=[T1_net, L_net])
        proj_net = MappingProjection(sender=T2_net, receiver=L_net.input_states[1])
        p2_net = Process(pathway=[T2_net, proj_net, L_net])
        s_net = System(processes=[p1_net, p2_net])
        for i in range(num_trials):
            out_spec = s_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1][0]
            leabra_output = run_network(leabra_net, inputs[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = s_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1][0]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)
        out_spec = s_spec.run(inputs={T1_spec: inputs, T2_spec: train_data})
        pnl_output_spec = np.array(out_spec[-1][0])
        for i in range(len(inputs)):
            leabra_output = np.array(run_network(leabra_net, inputs[i]))
        diffs_spec = np.abs(pnl_output_spec - leabra_output)
        out_net = s_net.run(inputs={T1_net: inputs, T2_net: train_data})
        pnl_output_net = np.array(out_net[-1][0])
        diffs_net = np.abs(pnl_output_net - leabra_output)
        assert all(diffs_spec < precision) and all(diffs_net < precision)

    def test_leabra_prec_with_train(self):
        in_size = 4
        out_size = 4
        num_hidden = 1
        num_trials = 4
        train = True
        inputs = [[0, 1, .5, -.2]] * num_trials
        train_data = [[.2, .5, 1, -.5]] * num_trials
        precision = 0.00001  # how far we accept error between PNL and Leabra output
        random_seed = 1  # because Leabra network initializes with small random weights
        random.seed(random_seed)
        L_spec = LeabraMechanism(input_size=in_size, output_size=out_size, hidden_layers=num_hidden, training_flag=train)
        random.seed(random_seed)
        leabra_net = build_leabra_network(in_size, out_size, num_hidden, None, train)
        leabra_net2 = copy.deepcopy(leabra_net)
        L_net = LeabraMechanism(leabra_net2)
        # leabra_net should be identical to the network inside L_net

        T1_spec = TransferMechanism(name='T1', size=in_size, function=Linear)
        T2_spec = TransferMechanism(name='T2', size=out_size, function=Linear)
        T1_net = TransferMechanism(name='T1', size=in_size, function=Linear)
        T2_net = TransferMechanism(name='T2', size=out_size, function=Linear)

        p1_spec = Process(pathway=[T1_spec, L_spec])
        proj_spec = MappingProjection(sender=T2_spec, receiver=L_spec.input_states[1])
        p2_spec = Process(pathway=[T2_spec, proj_spec, L_spec])
        s_spec = System(processes=[p1_spec, p2_spec])

        p1_net = Process(pathway=[T1_net, L_net])
        proj_net = MappingProjection(sender=T2_net, receiver=L_net.input_states[1])
        p2_net = Process(pathway=[T2_net, proj_net, L_net])
        s_net = System(processes=[p1_net, p2_net])
        for i in range(num_trials):
            out_spec = s_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1][0]
            leabra_output = train_leabra_network(leabra_net, inputs[i], train_data[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = s_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1][0]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)
        out_spec = s_spec.run(inputs={T1_spec: inputs, T2_spec: train_data})
        pnl_output_spec = np.array(out_spec[-1][0])
        for i in range(len(inputs)):
            leabra_output = np.array(train_leabra_network(leabra_net, inputs[i], train_data[i]))
        diffs_spec = np.abs(pnl_output_spec - leabra_output)
        out_net = s_net.run(inputs={T1_net: inputs, T2_net: train_data})
        pnl_output_net = np.array(out_net[-1][0])
        diffs_net = np.abs(pnl_output_net - leabra_output)
        assert all(diffs_spec < precision) and all(diffs_net < precision)

    # do one round of training, one round of non-training
    def test_leabra_prec_half_train(self):
        in_size = 4
        out_size = 4
        num_hidden = 1
        num_trials = 2
        train = True
        inputs = [[0, 1, .5, -.2]] * num_trials
        train_data = [[.2, .5, 1, -.5]] * num_trials
        precision = 0.00001  # how far we accept error between PNL and Leabra output
        random_seed = 1  # because Leabra network initializes with small random weights
        random.seed(random_seed)
        L_spec = LeabraMechanism(input_size=in_size, output_size=out_size, hidden_layers=num_hidden, training_flag=train)
        random.seed(random_seed)
        leabra_net = build_leabra_network(in_size, out_size, num_hidden, None, train)
        leabra_net2 = copy.deepcopy(leabra_net)
        L_net = LeabraMechanism(leabra_net2)
        # leabra_net should be identical to the network inside L_net

        T1_spec = TransferMechanism(name='T1', size=in_size, function=Linear)
        T2_spec = TransferMechanism(name='T2', size=out_size, function=Linear)
        T1_net = TransferMechanism(name='T1', size=in_size, function=Linear)
        T2_net = TransferMechanism(name='T2', size=out_size, function=Linear)

        p1_spec = Process(pathway=[T1_spec, L_spec])
        proj_spec = MappingProjection(sender=T2_spec, receiver=L_spec.input_states[1])
        p2_spec = Process(pathway=[T2_spec, proj_spec, L_spec])
        s_spec = System(processes=[p1_spec, p2_spec])

        p1_net = Process(pathway=[T1_net, L_net])
        proj_net = MappingProjection(sender=T2_net, receiver=L_net.input_states[1])
        p2_net = Process(pathway=[T2_net, proj_net, L_net])
        s_net = System(processes=[p1_net, p2_net])
        for i in range(num_trials):  # training round
            out_spec = s_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1][0]
            leabra_output = train_leabra_network(leabra_net, inputs[i], train_data[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = s_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1][0]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)

        L_net.training_flag = False
        L_spec.training_flag = False
        for i in range(num_trials):  # non-training round
            out_spec = s_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1][0]
            leabra_output = run_network(leabra_net, inputs[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = s_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1][0]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)
#
# class TestLeabraMechanismInSystem:
#
#     def test_leabra_mech_learning(self):
#         T1 = TransferMechanism(size=5, function=Linear)
#         T2 = TransferMechanism(size=3, function=Linear)
#         L = LeabraMechanism(input_size=5, output_size=3, hidden_layers=2, hidden_sizes=[4, 4])
#         train_data_proj = MappingProjection(sender=T2, receiver=L.input_states[1])
#         out = TransferMechanism(size=3, function=Logistic(bias=2))
#         p1 = Process(pathway=[T1, L, out], learning=LEARNING, learning_rate=1.0, target=[0, .1, .8])
#         p2 = Process(pathway=[T2, train_data_proj, L, out])
#         s = System(processes=[p1, p2])
#         s.run(inputs = {T1: [1, 2, 3, 4, 5], T2: [0, .5, 1]})