import copy
import numpy as np
import pytest
import random
try:
    import leabra
except ImportError:
    leabra = None

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.keywords import LEARNING
from psyneulink.library.components.mechanisms.processing.leabramechanism import LeabraMechanism, build_leabra_network, run_leabra_network, train_leabra_network

LEABRA_NOT_AVAILABLE='leabra python module is not installed. Please install it from https://github.com/benureau/leabra'

@pytest.mark.skipif(leabra is None, reason=LEABRA_NOT_AVAILABLE)
class TestLeabraMechInit:

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
        assert np.sum(np.abs(val)) <= 0.001

    # NOTE 11/3/17 CW: I have no intuition about what these values should be, so I'm not "testing" output values for now
    def test_leabra_init_no_hidden_sizes(self):
        L = LeabraMechanism(input_size=4, output_size=4, hidden_layers=2, training_flag=False)
        val = L.execute([[1, 2, 3, 4], [0, 0, 0, 0]])
        assert L.hidden_layers == 2
        assert L.hidden_sizes == 4
        assert len(val[0]) == 4


@pytest.mark.skipif(leabra is None, reason=LEABRA_NOT_AVAILABLE)
class TestLeabraMechRuntimeParams:

    def test_leabra_runtime_alone(self):
        n_input = 4
        n_output = 3
        n_hidden = 2
        hidden_sizes = None
        inputs = [[.1, .2, .3, .4], [.4, .5, .6, .7], [-.6, -.7, -.8, -.9]]
        train_input = [1, 0, -1]
        random.seed(10)
        L1 = LeabraMechanism(input_size=n_input, output_size=n_output, hidden_layers=n_hidden,
                             hidden_sizes=None, training_flag=False)  # training flag is false
        random.seed(10)
        L2 = LeabraMechanism(input_size=n_input, output_size=n_output, hidden_layers=n_hidden,
                             hidden_sizes=None, training_flag=True)  # training flag is true
        random.seed(10)
        net = build_leabra_network(n_input, n_output, n_hidden, hidden_sizes, False)

        pnl_output1_1 = L1.execute(input=[inputs[0], train_input], runtime_params={"training_flag": False})
        pnl_output2_1 = L2.execute(input=[inputs[0], train_input], runtime_params={"training_flag": False})
        net_output_1 = run_leabra_network(net, input_pattern=inputs[0])
        np.testing.assert_allclose(pnl_output1_1[0], net_output_1, atol=1e-08)
        np.testing.assert_allclose(pnl_output2_1[0], net_output_1, atol=1e-08)

        pnl_output1_2 = L1.execute(input=[inputs[1], train_input], runtime_params={"training_flag": True})
        pnl_output2_2 = L2.execute(input=[inputs[1], train_input], runtime_params={"training_flag": True})
        net_output_2 = train_leabra_network(net, input_pattern=inputs[1], output_pattern=train_input)
        np.testing.assert_allclose(pnl_output1_2[0], net_output_2, atol=1e-08)
        np.testing.assert_allclose(pnl_output2_2[0], net_output_2, atol=1e-08)

        pnl_output1_3 = L1.execute(input=[inputs[2], train_input], runtime_params={"training_flag": False})
        pnl_output2_3 = L2.execute(input=[inputs[2], train_input], runtime_params={"training_flag": False})
        net_output_3 = run_leabra_network(net, input_pattern=inputs[2])
        np.testing.assert_allclose(pnl_output1_3[0], net_output_3, atol=1e-08)
        np.testing.assert_allclose(pnl_output2_3[0], net_output_3, atol=1e-08)

    def test_leabra_runtime_in_system(self):
        pass


@pytest.mark.skipif(leabra is None, reason=LEABRA_NOT_AVAILABLE)
class TestLeabraMechPrecision:

    def test_leabra_prec_no_train(self):
        in_size = 4
        out_size = 4
        num_hidden = 1
        num_trials = 2
        train = False
        inputs = [[0, 1, -1, 2]] * num_trials
        train_data = [[10] * out_size] * num_trials
        precision = 0.000000001  # how far we accept error between PNL and Leabra output
        random_seed = 1  # because Leabra network initializes with small random weights
        random.seed(random_seed)
        L_spec = LeabraMechanism(input_size=in_size, output_size=out_size, hidden_layers=num_hidden, training_flag=train)
        random.seed(random_seed)
        leabra_net = build_leabra_network(in_size, out_size, num_hidden, None, train)
        leabra_net2 = copy.deepcopy(leabra_net)
        L_net = LeabraMechanism(leabra_net2)
        # leabra_net should be identical to the network inside L_net

        T1_spec = TransferMechanism(name='T1_spec', size=in_size, function=Linear)
        T2_spec = TransferMechanism(name='T2_spec', size=out_size, function=Linear)
        T1_net = TransferMechanism(name='T1_net', size=in_size, function=Linear)
        T2_net = TransferMechanism(name='T2_net', size=out_size, function=Linear)

        proj_spec = MappingProjection(sender=T2_spec, receiver=L_spec.input_ports[1])
        c_spec = Composition(pathways=[[T1_spec, L_spec],[T2_spec, proj_spec, L_spec]])

        proj_net = MappingProjection(sender=T2_net, receiver=L_net.input_ports[1])
        c_net = Composition(pathways=[[T1_net, L_net],[T2_net, proj_net, L_net]])

        for i in range(num_trials):
            out_spec = c_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1]
            leabra_output = run_leabra_network(leabra_net, inputs[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = c_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)
        out_spec = c_spec.run(inputs={T1_spec: inputs, T2_spec: train_data})
        pnl_output_spec = np.array(out_spec[-1])
        for i in range(len(inputs)):
            leabra_output = np.array(run_leabra_network(leabra_net, inputs[i]))
        diffs_spec = np.abs(pnl_output_spec - leabra_output)
        out_net = c_net.run(inputs={T1_net: inputs, T2_net: train_data})
        pnl_output_net = np.array(out_net[-1])
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
        precision = 0.000000001  # how far we accept error between PNL and Leabra output
        random_seed = 2  # because Leabra network initializes with small random weights
        random.seed(random_seed)
        L_spec = LeabraMechanism(input_size=in_size, output_size=out_size, hidden_layers=num_hidden,
                                 training_flag=train)
        random.seed(random_seed)
        leabra_net = build_leabra_network(in_size, out_size, num_hidden, None, train)
        leabra_net2 = copy.deepcopy(leabra_net)
        L_net = LeabraMechanism(leabra_net2)
        # leabra_net should be identical to the network inside L_net

        T1_spec = TransferMechanism(name='T1_spec', size=in_size, function=Linear)
        T2_spec = TransferMechanism(name='T2_spec', size=out_size, function=Linear)
        T1_net = TransferMechanism(name='T1_net', size=in_size, function=Linear)
        T2_net = TransferMechanism(name='T2_net', size=out_size, function=Linear)

        proj_spec = MappingProjection(sender=T2_spec, receiver=L_spec.input_ports[1])
        c_spec = Composition(pathways=[[T1_spec, L_spec],[T2_spec, proj_spec, L_spec]])

        proj_net = MappingProjection(sender=T2_net, receiver=L_net.input_ports[1])
        c_net = Composition(pathways=[[T1_net, L_net],[T2_net, proj_net, L_net]])

        for i in range(num_trials):
            out_spec = c_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1]
            leabra_output = train_leabra_network(leabra_net, inputs[i], train_data[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = c_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)
        out_spec = c_spec.run(inputs={T1_spec: inputs, T2_spec: train_data})
        pnl_output_spec = np.array(out_spec[-1])
        for i in range(len(inputs)):
            leabra_output = np.array(train_leabra_network(leabra_net, inputs[i], train_data[i]))
        diffs_spec = np.abs(pnl_output_spec - leabra_output)
        out_net = c_net.run(inputs={T1_net: inputs, T2_net: train_data})
        pnl_output_net = np.array(out_net[-1])
        diffs_net = np.abs(pnl_output_net - leabra_output)
        assert all(diffs_spec < precision) and all(diffs_net < precision)
        # assert np.sum(np.abs(pnl_output_spec - np.array(train_data[0]))) < 0.1
        # assert np.sum(np.abs(pnl_output_net - np.array(train_data[0]))) < 0.1

    # do one round of training, one round of non-training
    def test_leabra_prec_half_train(self):
        in_size = 4
        out_size = 4
        num_hidden = 1
        num_trials = 2
        train = True
        inputs = [[0, 1, .5, -.2]] * num_trials
        train_data = [[.2, .5, 1, -.5]] * num_trials
        precision = 0.000000001  # how far we accept error between PNL and Leabra output
        random_seed = 3  # because Leabra network initializes with small random weights
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

        proj_spec = MappingProjection(sender=T2_spec, receiver=L_spec.input_ports[1])
        c_spec = Composition(pathways=[[T1_spec, L_spec], [T2_spec, proj_spec, L_spec]])

        proj_net = MappingProjection(sender=T2_net, receiver=L_net.input_ports[1])
        c_net = Composition(pathways=[[T1_net, L_net],[T2_net, proj_net, L_net]])

        for i in range(num_trials):  # training round
            out_spec = c_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1]
            leabra_output = train_leabra_network(leabra_net, inputs[i], train_data[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = c_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)

        # assert np.sum(np.abs(pnl_output_spec - np.array(train_data[0]))) < 0.1
        # assert np.sum(np.abs(pnl_output_net - np.array(train_data[0]))) < 0.1

        # set all learning rules false
        for conn in leabra_net.connections:
            conn.spec.lrule = None
        L_net.parameters.training_flag.set(False, c_net)
        L_spec.parameters.training_flag.set(False, c_spec)

        for i in range(num_trials):  # non-training round
            out_spec = c_spec.run(inputs={T1_spec: inputs[i], T2_spec: train_data[i]})
            pnl_output_spec = out_spec[-1]
            leabra_output = run_leabra_network(leabra_net, inputs[i])
            diffs_spec = np.abs(np.array(pnl_output_spec) - np.array(leabra_output))
            out_net = c_net.run(inputs={T1_net: inputs[i], T2_net: train_data[i]})
            pnl_output_net = out_net[-1]
            diffs_net = np.abs(np.array(pnl_output_net) - np.array(leabra_output))
            assert all(diffs_spec < precision) and all(diffs_net < precision)

# class TestLeabraMechInSystem:
#
#     def test_leabra_mech_learning(self):
#         T1 = TransferMechanism(size=5, function=Linear)
#         T2 = TransferMechanism(size=3, function=Linear)
#         L = LeabraMechanism(input_size=5, output_size=3, hidden_layers=2, hidden_sizes=[4, 4])
#         train_data_proj = MappingProjection(sender=T2, receiver=L.input_ports[1])
#         out = TransferMechanism(size=3, function=Logistic(bias=2))
#         p1 = Process(pathway=[T1, L, out], learning=LEARNING, learning_rate=1.0, target=[0, .1, .8])
#         p2 = Process(pathway=[T2, train_data_proj, L, out])
#         s = System(processes=[p1, p2])
#         s.run(inputs = {T1: [1, 2, 3, 4, 5], T2: [0, .5, 1]})
