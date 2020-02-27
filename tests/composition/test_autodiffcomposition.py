import logging
import timeit as timeit

import numpy as np

import pytest

import psyneulink as pnl

from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals import Context
from psyneulink.core.globals.keywords import TRAINING_SET
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.system import System
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition

logger = logging.getLogger(__name__)


# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for functions of AutodiffComposition class that are new (not in Composition)
# or override functions in Composition


@pytest.mark.pytorch
@pytest.mark.acconstructor
class TestACConstructor:

    def test_no_args(self):
        comp = AutodiffComposition()
        assert isinstance(comp, AutodiffComposition)

    def test_two_calls_no_args(self):
        comp = AutodiffComposition()
        comp_2 = AutodiffComposition()
        assert isinstance(comp, AutodiffComposition)
        assert isinstance(comp_2, AutodiffComposition)

    # KAM removed this pytest 10/30 after removing target_CIM
    # def test_target_CIM(self):
    #     comp = AutodiffComposition()
    #     assert isinstance(comp.target_CIM, CompositionInterfaceMechanism)
    #     assert comp.target_CIM.composition == comp
    #     assert comp.target_CIM_ports == {}

    def test_pytorch_representation(self):
        comp = AutodiffComposition()
        assert comp.pytorch_representation is None

    def test_report_prefs(self):
        comp = AutodiffComposition()
        assert comp.input_CIM.reportOutputPref == False
        assert comp.output_CIM.reportOutputPref == False
        # assert comp.target_CIM.reportOutputPref == False
    
    # FIXME: This test for patience doesn't actually test for correctness
    # def test_patience(self):
        # comp = AutodiffComposition()
        # assert comp.patience == 10


@pytest.mark.pytorch
@pytest.mark.acmisc
class TestMiscTrainingFunctionality:

    # test whether pytorch parameters are initialized to be identical to the Autodiff Composition's
    # projections when AC is initialized with the "param_init_from_pnl" argument set to True
    def test_param_init_from_pnl(self):

        # create xor model mechanisms and projections
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection(matrix=np.random.rand(2,10))
        out_map = MappingProjection(matrix=np.random.rand(10,1))

        # put the mechanisms and projections together in an autodiff composition (AC)
        xor = AutodiffComposition(param_init_from_pnl=True)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        # mini version of xor.execute just to build up pytorch representation
        xor._analyze_graph()
        xor._build_pytorch_representation(context=xor.default_execution_id)
        # check whether pytorch parameters are identical to projections
        assert np.allclose(hid_map.parameters.matrix.get(None),
                           xor.parameters.pytorch_representation.get(xor).params[0].detach().numpy())
        assert np.allclose(out_map.parameters.matrix.get(None),
                           xor.parameters.pytorch_representation.get(xor).params[1].detach().numpy())

    # test whether processing doesn't interfere with pytorch parameters after training
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    def test_training_then_processing(self, mode):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection()
        out_map = MappingProjection()

        xor = AutodiffComposition(param_init_from_pnl=True)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        
        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        # train model for a few epochs
        # results_before_proc = xor.run(inputs={xor_in:xor_inputs},
        #                               targets={xor_out:xor_targets},
        #                               epochs=10)
        results_before_proc = xor.learn(inputs={"inputs": {xor_in:xor_inputs},
                                              "targets": {xor_out:xor_targets},
                                              "epochs": 10}, bin_execute=mode)

        # get weight parameters from pytorch
        pt_weights_hid_bp = xor.parameters.pytorch_representation.get(xor).params[0].detach().numpy().copy()
        pt_weights_out_bp = xor.parameters.pytorch_representation.get(xor).params[1].detach().numpy().copy()

        #KAM temporarily removed -- will reimplement when pytorch weights can be used in pure PNL execution
        # do processing on a few inputs
        # results_proc = xor.run(inputs={xor_in:xor_inputs})
        # results_proc = xor.run(inputs={"inputs": {xor_in:xor_inputs}})
        #
        # # get weight parameters from pytorch
        # pt_weights_hid_ap = xor.parameters.pytorch_representation.get(xor).params[0].detach().numpy().copy()
        # pt_weights_out_ap = xor.parameters.pytorch_representation.get(xor).params[1].detach().numpy().copy()
        #
        # # check that weight parameters before and after processing are the same
        # assert np.allclose(pt_weights_hid_bp, pt_weights_hid_ap)
        # assert np.allclose(pt_weights_out_bp, pt_weights_out_ap)

    @pytest.mark.parametrize(
        'loss', ['l1', 'poissonnll']
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=[pytest.mark.llvm,pytest.mark.skip]), # these loss specs remain unimplemented at the moment
                                     ])
    def test_various_loss_specs(self, loss, mode):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection()
        out_map = MappingProjection()

        xor = AutodiffComposition(param_init_from_pnl=True, loss_spec=loss)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        xor.learn(inputs = {"inputs": {xor_in:xor_inputs},
                          "targets": {xor_out:xor_targets},
                          "epochs": 10}, bin_execute=mode)

    @pytest.mark.parametrize("mode", ['Python',
                                    #   pytest.param('LLVMRun', marks=[pytest.mark.llvm, pytest.mark.skip]), # Not implemented?
                                     ])
    def test_pytorch_loss_spec(self, mode):
        import torch
        ls = torch.nn.SoftMarginLoss(reduction='sum')

        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection()
        out_map = MappingProjection()

        xor = AutodiffComposition(param_init_from_pnl=True, loss_spec=ls)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0], [0, 1], [1, 0], [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0], [1], [1], [0]])

        xor.learn(inputs={"inputs": {xor_in:xor_inputs},
                        "targets": {xor_out:xor_targets},
                        "epochs": 10}, bin_execute=mode)
        xor.learn(inputs={"inputs": {xor_in: xor_inputs},
                        "targets": {xor_out: xor_targets},
                        "epochs": 10}, bin_execute=mode)


    @pytest.mark.benchmark(group="Optimizer specs")
    @pytest.mark.parametrize(
        'learning_rate, weight_decay, optimizer_type', [
            (10, 0, 'sgd'), (1.5, 1, 'sgd'),  (1.5, 1, 'adam'),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    def test_optimizer_specs(self, learning_rate, weight_decay, optimizer_type, mode, benchmark):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection()
        out_map = MappingProjection()

        xor = AutodiffComposition(param_init_from_pnl=True,
                                  learning_rate=learning_rate,
                                  optimizer_type=optimizer_type,
                                  weight_decay=weight_decay)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0], [0, 1], [1, 0], [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0], [1], [1], [0]])

        # train model for a few epochs
        # results_before_proc = xor.run(inputs={xor_in:xor_inputs},
        #                               targets={xor_out:xor_targets},
        #                               epochs=10)
        results_before_proc = xor.learn(inputs={"inputs": {xor_in:xor_inputs},
                                              "targets": {xor_out:xor_targets},
                                              "epochs": 10}, bin_execute=mode)

        benchmark(xor.learn, inputs={"inputs": {xor_in:xor_inputs},
                                   "targets": {xor_out:xor_targets},
                                   "epochs": 10}, bin_execute=mode)


    # test whether pytorch parameters and projections are kept separate (at diff. places in memory)
    @pytest.mark.parametrize("mode", ['Python',
                                    #   pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                    #   LLVM test is disabled since weights are always copied back
                                     ])
    def test_params_stay_separate(self,mode):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_m = np.random.rand(2,10)
        out_m = np.random.rand(10,1)

        hid_map = MappingProjection(name='hid_map',
                                    matrix=hid_m.copy(),
                                    sender=xor_in,
                                    receiver=xor_hid)

        out_map = MappingProjection(name='out_map',
                                    matrix=out_m.copy(),
                                    sender=xor_hid,
                                    receiver=xor_out)

        xor = AutodiffComposition(param_init_from_pnl=True,
                                  learning_rate=10.0,
                                  optimizer_type="sgd")

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0], [0, 1], [1, 0], [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0], [1], [1], [0]])

        # train the model for a few epochs
        result = xor.learn(inputs={"inputs": {xor_in:xor_inputs},
                                 "targets": {xor_out:xor_targets},
                                 "epochs": 10}, bin_execute=mode)

        # get weight parameters from pytorch
        pt_weights_hid = xor.parameters.pytorch_representation.get(xor).params[0].detach().numpy().copy()
        pt_weights_out = xor.parameters.pytorch_representation.get(xor).params[1].detach().numpy().copy()

        # assert that projections are still what they were initialized as
        assert np.allclose(hid_map.parameters.matrix.get(None), hid_m)
        assert np.allclose(out_map.parameters.matrix.get(None), out_m)

        # assert that projections didn't change during training with the pytorch
        # parameters (they should now be different)
        assert not np.allclose(pt_weights_hid, hid_map.parameters.matrix.get(None))
        assert not np.allclose(pt_weights_out, out_map.parameters.matrix.get(None))

    # test whether the autodiff composition's get_parameters method works as desired
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                    #   LLVM test is disabled since parameters are currently not written back

                                     ])
    def test_get_params(self, mode):

        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection(matrix=np.random.rand(2,10))
        out_map = MappingProjection(matrix=np.random.rand(10,1))

        xor = AutodiffComposition(param_init_from_pnl=True,
                                  learning_rate=1.0)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0], [0, 1], [1, 0], [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0], [1], [1], [0]])

        # call run to only process the inputs, so that pytorch representation of AC gets created
        # results = xor.run(inputs={xor_in:xor_inputs})

        #KAM Changed 11/1/18

        # mini version of xor.execute just to build up pytorch representation
        xor._analyze_graph()
        # CW changed 12/3/18
        xor._build_pytorch_representation(xor.default_execution_id)
        # OLD
        # xor._build_pytorch_representation()

        # call get_parameters to obtain a copy of the pytorch parameters in numpy arrays,
        # and get the parameters straight from pytorch
        weights_get_params = xor.get_parameters()
        weights_straight_1 = xor.parameters.pytorch_representation.get(xor).params[0]
        weights_straight_2 = xor.parameters.pytorch_representation.get(xor).params[1]

        # check that parameter copies obtained from get_parameters are the same as the
        # projections and parameters from pytorch
        assert np.allclose(hid_map.parameters.matrix.get(None), weights_get_params[hid_map])
        assert np.allclose(weights_straight_1.detach().numpy(), weights_get_params[hid_map])
        assert np.allclose(out_map.parameters.matrix.get(None), weights_get_params[out_map])
        assert np.allclose(weights_straight_2.detach().numpy(), weights_get_params[out_map])

        # call run to train the pytorch parameters
        results = xor.learn(inputs={"inputs": {xor_in:xor_inputs},
                                  "targets": {xor_out:xor_targets},
                                  "epochs": 10}, bin_execute=mode)


        # check that the parameter copies obtained from get_parameters have not changed with the
        # pytorch parameters during training (and are thus at a different memory location)
        # (only makes sense in Python mode)
        if mode == 'Python':
            assert not np.allclose(weights_straight_1.detach().numpy(), weights_get_params[hid_map])
            assert not np.allclose(weights_straight_2.detach().numpy(), weights_get_params[out_map])


@pytest.mark.pytorch
@pytest.mark.accorrectness
class TestTrainingCorrectness:

    # test whether xor model created as autodiff composition learns properly
    @pytest.mark.benchmark(group="XOR")
    @pytest.mark.parametrize(
        'eps, calls, opt, from_pnl_or_not', [
            (50, 'single', 'adam', True),
            # (50, 'multiple', 'adam', True),
            (50, 'single', 'adam', False),
            # (50, 'multiple', 'adam', False)
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    def test_xor_training_correctness(self, eps, calls, opt, from_pnl_or_not, mode, benchmark):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection(matrix=np.random.rand(2,10), sender=xor_in, receiver=xor_hid)
        out_map = MappingProjection(matrix=np.random.rand(10,1))

        xor = AutodiffComposition(param_init_from_pnl=from_pnl_or_not,
                                  optimizer_type=opt,
                                  learning_rate=0.1)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0], [0, 1], [1, 0], [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0], [1], [1], [0]])

        if calls == 'single':
            results = xor.learn(inputs={"inputs": {xor_in:xor_inputs},
                                      "targets": {xor_out:xor_targets},
                                      "epochs": eps}, bin_execute=mode)

            for i in range(len(results[0])):
                assert np.allclose(np.round(results[0][i][0]), xor_targets[i])

        else:
            results = xor.learn(inputs={"inputs": {xor_in: xor_inputs},
                                      "targets": {xor_out: xor_targets},
                                      "epochs": 1}, bin_execute=mode)

            for i in range(eps - 1):
                results = xor.learn(inputs={"inputs": {xor_in: xor_inputs},
                                          "targets": {xor_out: xor_targets},
                                          "epochs": 1}, bin_execute=mode)

            for i in range(len(results[eps - 1])):
                assert np.allclose(np.round(results[eps - 1][i][0]), xor_targets[i])

        benchmark(xor.learn, inputs={'inputs': {xor_in: xor_inputs},
                                   'targets': {xor_out: xor_targets},
                                   'epochs': eps}, bin_execute=mode)


    # tests whether semantic network created as autodiff composition learns properly
    @pytest.mark.benchmark(group="Semantic net")
    @pytest.mark.parametrize(
        'eps, opt, from_pnl_or_not', [
            (500, 'adam', True),
            # (300, 'adam', False)
        ]
    )
    @pytest.mark.parametrize("mode", ["Python",
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    def test_semantic_net_training_correctness(self, eps, opt, from_pnl_or_not, mode, benchmark):

        # MECHANISMS FOR SEMANTIC NET:

        nouns_in = TransferMechanism(name="nouns_input",
                                     default_variable=np.zeros(8))

        rels_in = TransferMechanism(name="rels_input",
                                    default_variable=np.zeros(3))

        h1 = TransferMechanism(name="hidden_nouns",
                               default_variable=np.zeros(8),
                               function=Logistic())

        h2 = TransferMechanism(name="hidden_mixed",
                               default_variable=np.zeros(15),
                               function=Logistic())

        out_sig_I = TransferMechanism(name="sig_outs_I",
                                      default_variable=np.zeros(8),
                                      function=Logistic())

        out_sig_is = TransferMechanism(name="sig_outs_is",
                                       default_variable=np.zeros(12),
                                       function=Logistic())

        out_sig_has = TransferMechanism(name="sig_outs_has",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        out_sig_can = TransferMechanism(name="sig_outs_can",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        # SET UP PROJECTIONS FOR SEMANTIC NET

        map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                         name="map_nouns_h1",
                                         sender=nouns_in,
                                         receiver=h1)

        map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                        name="map_relh2",
                                        sender=rels_in,
                                        receiver=h2)

        map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                                      name="map_h1_h2",
                                      sender=h1,
                                      receiver=h2)

        map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                                     name="map_h2_I",
                                    sender=h2,
                                    receiver=out_sig_I)

        map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                                      name="map_h2_is",
                                      sender=h2,
                                      receiver=out_sig_is)

        map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_has",
                                       sender=h2,
                                       receiver=out_sig_has)

        map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_can",
                                       sender=h2,
                                       receiver=out_sig_can)

        # COMPOSITION FOR SEMANTIC NET
        sem_net = AutodiffComposition(param_init_from_pnl=from_pnl_or_not,
                                      optimizer_type=opt, learning_rate=.001)

        sem_net.add_node(nouns_in)
        sem_net.add_node(rels_in)
        sem_net.add_node(h1)
        sem_net.add_node(h2)
        sem_net.add_node(out_sig_I)
        sem_net.add_node(out_sig_is)
        sem_net.add_node(out_sig_has)
        sem_net.add_node(out_sig_can)

        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)

        # INPUTS & OUTPUTS FOR SEMANTIC NET:

        nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
        relations = ['is', 'has', 'can']
        is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
                   'yellow']
        has_list = ['roots', 'leaves', 'bark', 'branches', 'skin', 'feathers', 'wings', 'gills', 'scales']
        can_list = ['grow', 'move', 'swim', 'fly', 'breathe', 'breathe underwater', 'breathe air', 'walk', 'photosynthesize']

        nouns_input = np.identity(len(nouns))

        rels_input = np.identity(len(relations))

        truth_nouns = np.identity(len(nouns))

        truth_is = np.zeros((len(nouns), len(is_list)))

        truth_is[0, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
        truth_is[1, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
        truth_is[2, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        truth_is[3, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        truth_is[4, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[5, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
        truth_is[6, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        truth_is[7, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

        truth_has = np.zeros((len(nouns), len(has_list)))

        truth_has[0, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[1, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[2, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[3, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[4, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        truth_has[7, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]

        truth_can = np.zeros((len(nouns), len(can_list)))

        truth_can[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[3, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[4, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[5, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[6, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        truth_can[7, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]

        # SETTING UP DICTIONARY OF INPUTS/OUTPUTS FOR SEMANTIC NET

        inputs_dict = {}
        inputs_dict[nouns_in] = []
        inputs_dict[rels_in] = []

        targets_dict = {}
        targets_dict[out_sig_I] = []
        targets_dict[out_sig_is] = []
        targets_dict[out_sig_has] = []
        targets_dict[out_sig_can] = []

        for i in range(len(nouns)):
            for j in range(len(relations)):
                inputs_dict[nouns_in].append(nouns_input[i])
                inputs_dict[rels_in].append(rels_input[j])
                targets_dict[out_sig_I].append(truth_nouns[i])
                targets_dict[out_sig_is].append(truth_is[i])
                targets_dict[out_sig_has].append(truth_has[i])
                targets_dict[out_sig_can].append(truth_can[i])

        # TRAIN THE MODEL
        result = sem_net.learn(inputs={'inputs': inputs_dict,
                                      'targets': targets_dict,
                                      'epochs': eps}, bin_execute=mode)

        # CHECK CORRECTNESS
        for i in range(len(result)): # go over trial outputs in the single results entry
            for j in range(len(result[i])): # go over outputs for each output layer

                # get target for terminal node whose OutputPort corresponds to current output
                correct_value = None
                curr_CIM_input_port = sem_net.output_CIM.input_ports[j]
                for output_port in sem_net.output_CIM_ports.keys():
                    if sem_net.output_CIM_ports[output_port][0] == curr_CIM_input_port:
                        node = output_port.owner
                        correct_value = targets_dict[node][i]

                # compare model output for terminal node on current trial with target for terminal node on current trial
                assert np.allclose(np.round(result[i][j]), correct_value)

        benchmark(sem_net.learn, inputs={'inputs': inputs_dict,
                                       'targets': targets_dict,
                                       'epochs': eps}, bin_execute=mode)

    @pytest.mark.parametrize("mode", ["Python",
                                pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                ])
    def test_pytorch_equivalence_with_autodiff_composition(self, mode):
        iSs = np.array(
            [np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996])]
        )

        cSs = np.array(
            [np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 1.])]
        )

        oSs = np.array(
            [np.array([0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([1., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 1., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., -0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])]
        )

        nf = 3
        nd = 5
        nh = 200

        D_i = nf * nd
        D_c = nd ** 2
        D_h = nh
        D_o = nf * nd

        wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
        wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
        wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
        who = np.random.rand(D_h, D_o) * 0.02 - 0.01

        patience = 10
        min_delt = 0.00001
        learning_rate = 100

        il = TransferMechanism(size=D_i, name='input')
        cl = TransferMechanism(size=D_c, name='control')
        hl = TransferMechanism(size=D_h, name='hidden',
                               function=Logistic(bias=-2))
        ol = TransferMechanism(size=D_o, name='output',
                               function=Logistic(bias=-2))

        input_set = {
            'inputs': {
                il: iSs,
                cl: cSs
            },
            'targets': {
                ol: oSs
            }
        }

        pih = MappingProjection(matrix=wih)
        pch = MappingProjection(matrix=wch)
        pco = MappingProjection(matrix=wco)
        pho = MappingProjection(matrix=who)

        mnet = AutodiffComposition(param_init_from_pnl=True,
                                   learning_rate=learning_rate)

        mnet.add_node(il)
        mnet.add_node(cl)
        mnet.add_node(hl)
        mnet.add_node(ol)
        mnet.add_projection(projection=pih, sender=il, receiver=hl)
        mnet.add_projection(projection=pch, sender=cl, receiver=hl)
        mnet.add_projection(projection=pco, sender=cl, receiver=ol)
        mnet.add_projection(projection=pho, sender=hl, receiver=ol)

        mnet.learn(
            inputs=input_set,
            minibatch_size=1,
            patience=patience,
            min_delta=min_delt,
            bin_execute=mode
        )
        mnet.run(
            inputs=input_set['inputs'],
            bin_execute=mode
        )
        output = np.array(mnet.parameters.results.get(mnet)[-15:]).reshape(225)

        comparator = np.array([0.02288846, 0.11646781, 0.03473711, 0.0348004, 0.01679579,
                             0.04851733, 0.05857743, 0.04819957, 0.03004438, 0.05113508,
                             0.06849843, 0.0442623, 0.00967315, 0.06998125, 0.03482444,
                             0.05856816, 0.00724313, 0.03676571, 0.03668758, 0.01761947,
                             0.0516829, 0.06260267, 0.05160782, 0.03140498, 0.05462971,
                             0.07360401, 0.04687923, 0.00993319, 0.07662302, 0.03687142,
                             0.0056837, 0.03411045, 0.03615285, 0.03606166, 0.01774354,
                             0.04700402, 0.09696857, 0.06843472, 0.06108671, 0.0485631,
                             0.07194324, 0.04485926, 0.00526768, 0.07442083, 0.0364541,
                             0.02819926, 0.03804169, 0.04091214, 0.04091113, 0.04246229,
                             0.05583883, 0.06643675, 0.05630667, 0.01540373, 0.05948422,
                             0.07721549, 0.05081813, 0.01205326, 0.07998289, 0.04084186,
                             0.02859247, 0.03794089, 0.04111452, 0.04139213, 0.01222424,
                             0.05677404, 0.06736114, 0.05614553, 0.03573626, 0.05983103,
                             0.07867571, 0.09971621, 0.01203033, 0.08107789, 0.04110497,
                             0.02694072, 0.03592752, 0.03878366, 0.03895513, 0.01852774,
                             0.05097689, 0.05753834, 0.05090328, 0.03405996, 0.05293719,
                             0.07037981, 0.03474316, 0.02861534, 0.12504038, 0.0387827,
                             0.02467716, 0.03373265, 0.03676382, 0.03677551, 0.00758558,
                             0.089832, 0.06330426, 0.0514472, 0.03120581, 0.05535174,
                             0.07494839, 0.04169744, 0.00698747, 0.0771042, 0.03659954,
                             0.03008443, 0.0393799, 0.0423592, 0.04237004, 0.00965198,
                             0.09863199, 0.06813933, 0.05675321, 0.03668943, 0.0606036,
                             0.07898065, 0.04662618, 0.00954765, 0.08093391, 0.04218842,
                             0.02701085, 0.03660227, 0.04058368, 0.04012464, 0.02030738,
                             0.047633, 0.06693405, 0.055821, 0.03456592, 0.10166267,
                             0.07870758, 0.04935871, 0.01065449, 0.08012213, 0.04036544,
                             0.02576563, 0.03553382, 0.03920509, 0.03914452, 0.01907667,
                             0.05106766, 0.06555857, 0.05434728, 0.03335726, 0.05074808,
                             0.07715102, 0.04839309, 0.02494798, 0.08001304, 0.03921895,
                             0.00686952, 0.03941704, 0.04128484, 0.04117602, 0.02217508,
                             0.05152296, 0.10361618, 0.07488737, 0.0707186, 0.05289282,
                             0.07557573, 0.04978292, 0.00705783, 0.07787788, 0.04164007,
                             0.00574239, 0.03437231, 0.03641445, 0.03631848, 0.01795791,
                             0.04723996, 0.09732232, 0.06876138, 0.06156679, 0.04878423,
                             0.07214104, 0.04511085, 0.00535038, 0.07459818, 0.0367153,
                             0.02415251, 0.03298647, 0.03586635, 0.0360273, 0.01624523,
                             0.04829838, 0.05523439, 0.04821285, 0.03115052, 0.05034625,
                             0.06836408, 0.03264844, 0.0241706, 0.12190507, 0.03585727,
                             0.02897192, 0.03925683, 0.04250414, 0.04253885, 0.02175426,
                             0.05683923, 0.06547528, 0.05705267, 0.03742978, 0.05951711,
                             0.12675475, 0.05216411, 0.00181494, 0.08218002, 0.04234364,
                             0.02789848, 0.036924, 0.03976586, 0.03993866, 0.01932489,
                             0.05186586, 0.05829845, 0.05179337, 0.03504668, 0.05379566,
                             0.07103772, 0.03544133, 0.03019486, 0.12605846, 0.03976812])

        assert np.allclose(output,comparator)

    def test_pytorch_equivalence_with_autodiff_training_disabled_on_proj(self):
        iSs = np.array(
                [np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                           0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                           0.60783064, 0.32504722, 0.58185035, 0.4143686, 0.4746975]),
                 np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                           0.1059076, 0.21655035, 0.13521817, 0.324141, 0.65314,
                           0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
                 np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                           0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.65314,
                           0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
                 np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                           0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                           0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
                 np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                           0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                           0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
                 np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                           0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                           0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
                 np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                           0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                           0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
                 np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                           0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                           0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
                 np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                           0.1059076, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                           0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
                 np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                           0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.9023486,
                           0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
                 np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                           0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                           0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996]),
                 np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                           0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                           0.60783064, 0.32504722, 0.03842543, 0.63427407, 0.95894927]),
                 np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                           0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                           0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
                 np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                           0.73691815, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                           0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
                 np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                           0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                           0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996])]
        )

        cSs = np.array(
                [np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 1., 0., 0.]),
                 np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 1.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 1., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 1., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 1.]),
                 np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 1.])]
        )

        oSs = np.array(
                [np.array([0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([1., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 1., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
                 np.array([0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., -0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
                 np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])]
        )

        nf = 3
        nd = 5
        nh = 200

        D_i = nf * nd
        D_c = nd ** 2
        D_h = nh
        D_o = nf * nd

        np.random.seed(0)

        wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
        wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
        wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
        who = np.random.rand(D_h, D_o) * 0.02 - 0.01

        patience = 10
        min_delt = 0.00001
        learning_rate = 100

        il = TransferMechanism(size=D_i, name='input')
        cl = TransferMechanism(size=D_c, name='control')
        hl = TransferMechanism(size=D_h, name='hidden',
                               function=Logistic(bias=-2))
        ol = TransferMechanism(size=D_o, name='output',
                               function=Logistic(bias=-2))

        input_set = {
            'inputs': {
                il: iSs,
                cl: cSs
            },
            'targets': {
                ol: oSs
            }
        }

        pih = MappingProjection(matrix=wih)
        pch = MappingProjection(matrix=wch)
        pco = MappingProjection(matrix=wco)
        pho = MappingProjection(matrix=who, learnable=False)

        mnet = AutodiffComposition(param_init_from_pnl=True,
                                   learning_rate=learning_rate)

        mnet.add_node(il)
        mnet.add_node(cl)
        mnet.add_node(hl)
        mnet.add_node(ol)
        mnet.add_projection(projection=pih, sender=il, receiver=hl)
        mnet.add_projection(projection=pch, sender=cl, receiver=hl)
        mnet.add_projection(projection=pco, sender=cl, receiver=ol)
        mnet.add_projection(projection=pho, sender=hl, receiver=ol)

        
        mnet.learn(
                inputs=input_set,
                minibatch_size=1,
                patience=patience,
                min_delta=min_delt
        )

        print(mnet.parameters.results.get(mnet))
        mnet.run(
                inputs=input_set['inputs'],
        )

        output = np.array(mnet.parameters.results.get(mnet)[-15:]).reshape(225)

        comparator = np.array([0.10284232, 0.31514028, 0.10299414, 0.10164745, 0.10363132,
                               0.10164711, 0.10305342, 0.10162935, 0.10363974, 0.10175142,
                               0.10256631, 0.10194203, 0.10386363, 0.10445295, 0.10228054,
                               0.31140432, 0.10257346, 0.10279541, 0.1015088, 0.10408029,
                               0.10167408, 0.10260046, 0.10208146, 0.10258093, 0.10188455,
                               0.10239721, 0.10162553, 0.10376681, 0.10523887, 0.10231788,
                               0.08327345, 0.08337342, 0.0835557, 0.0828431, 0.08364569,
                               0.08285296, 0.21721269, 0.15223454, 0.12355195, 0.08328209,
                               0.08321026, 0.08318614, 0.08401372, 0.08443127, 0.08355132,
                               0.10225081, 0.10250866, 0.1032809, 0.10216374, 0.3212671,
                               0.10171002, 0.10223842, 0.10279202, 0.10348979, 0.102771,
                               0.10200755, 0.10137874, 0.10408875, 0.10449553, 0.10241774,
                               0.10293344, 0.10201894, 0.10300561, 0.10239243, 0.10291971,
                               0.10242151, 0.10280451, 0.10199619, 0.10344362, 0.10265052,
                               0.1030072, 0.31077573, 0.10299222, 0.10510338, 0.10226066,
                               0.08338644, 0.08334018, 0.08376527, 0.08334996, 0.08397464,
                               0.08293792, 0.08313457, 0.08310839, 0.08409815, 0.08289795,
                               0.08348748, 0.08323742, 0.35242194, 0.22024544, 0.08337309,
                               0.09164643, 0.09135997, 0.09195332, 0.09117354, 0.15678808,
                               0.25366357, 0.09192788, 0.09090009, 0.09173747, 0.09161069,
                               0.09198699, 0.09058125, 0.09191367, 0.09321109, 0.09121469,
                               0.09163069, 0.09134816, 0.09194396, 0.09114014, 0.15678652,
                               0.2536617, 0.09192093, 0.09089337, 0.09171399, 0.09160125,
                               0.09198645, 0.09058312, 0.09191372, 0.09321296, 0.09118975,
                               0.10222919, 0.1017347, 0.10354281, 0.10158797, 0.1038858,
                               0.10181702, 0.10269418, 0.10235615, 0.10275149, 0.31305784,
                               0.1030191, 0.10225646, 0.10283817, 0.10411466, 0.10244074,
                               0.10203665, 0.10201294, 0.10314981, 0.10192659, 0.10328009,
                               0.10265024, 0.1021864, 0.10181551, 0.1026119, 0.10268809,
                               0.10219657, 0.10172481, 0.32032955, 0.104648, 0.10248389,
                               0.08325538, 0.08334755, 0.08355319, 0.08281158, 0.08365688,
                               0.08285309, 0.21719442, 0.15221967, 0.12351983, 0.08326486,
                               0.08321615, 0.08318119, 0.08400558, 0.0844217, 0.08352901,
                               0.08326998, 0.08336743, 0.08356269, 0.08283862, 0.08365061,
                               0.08286179, 0.21723635, 0.15221801, 0.12355236, 0.08327687,
                               0.08322325, 0.08318282, 0.08401041, 0.08442231, 0.0835505,
                               0.0833958, 0.08335006, 0.08376891, 0.08336972, 0.08397432,
                               0.08294199, 0.08314709, 0.08311359, 0.0841146, 0.08291036,
                               0.08349533, 0.08323479, 0.35241473, 0.22023965, 0.08338647,
                               0.10243648, 0.10270733, 0.10287204, 0.10181676, 0.10309494,
                               0.10208003, 0.10258352, 0.10279328, 0.10355093, 0.10241994,
                               0.31674582, 0.10140157, 0.10286999, 0.10426361, 0.1018871,
                               0.08337424, 0.08333415, 0.08376191, 0.08333433, 0.08398008,
                               0.08293781, 0.08313539, 0.08310112, 0.08409653, 0.08289441,
                               0.08348761, 0.08323367, 0.35237628, 0.22024095, 0.08336799])

        assert np.allclose(output, comparator)


@pytest.mark.pytorch
@pytest.mark.actime
class TestTrainingTime:

    @pytest.mark.skip
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd'),
            (10, 'sgd'),
            (100, 'sgd')
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                    pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                    ])
    def test_and_training_time(self, eps, opt,mode):

        # SET UP MECHANISMS FOR COMPOSITION

        and_in = TransferMechanism(name='and_in',
                                   default_variable=np.zeros(2))

        and_out = TransferMechanism(name='and_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        # SET UP MECHANISMS FOR SYSTEM

        and_in_sys = TransferMechanism(name='and_in_sys',
                                       default_variable=np.zeros(2))

        and_out_sys = TransferMechanism(name='and_out_sys',
                                        default_variable=np.zeros(1),
                                        function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        and_map = MappingProjection(name='and_map',
                                    matrix=np.random.rand(2, 1),
                                    sender=and_in,
                                    receiver=and_out)

        # SET UP PROJECTIONS FOR SYSTEM

        and_map_sys = MappingProjection(name='and_map_sys',
                                        matrix=and_map.matrix.copy(),
                                        sender=and_in_sys,
                                        receiver=and_out_sys)

        # SET UP COMPOSITION

        and_net = AutodiffComposition(param_init_from_pnl=True)

        and_net.add_node(and_in)
        and_net.add_node(and_out)

        and_net.add_projection(sender=and_in, projection=and_map, receiver=and_out)

        # SET UP INPUTS AND TARGETS

        and_inputs = np.zeros((4,2))
        and_inputs[0] = [0, 0]
        and_inputs[1] = [0, 1]
        and_inputs[2] = [1, 0]
        and_inputs[3] = [1, 1]

        and_targets = np.zeros((4,1))
        and_targets[0] = [0]
        and_targets[1] = [1]
        and_targets[2] = [1]
        and_targets[3] = [0]

        # TIME TRAINING FOR COMPOSITION

        start = timeit.default_timer()
        result = and_net.run(inputs={and_in:and_inputs},
                             targets={and_out:and_targets},
                             epochs=eps,
                             learning_rate=0.1,
                             controller=opt,
                             bin_execute=mode)
        end = timeit.default_timer()
        comp_time = end - start

        # SET UP SYSTEM

        and_process = Process(pathway=[and_in_sys,
                                       and_map_sys,
                                       and_out_sys],
                              learning=pnl.LEARNING)

        and_sys = System(processes=[and_process],
                         learning_rate=0.1)

        # TIME TRAINING FOR SYSTEM

        start = timeit.default_timer()
        results_sys = and_sys.run(inputs={and_in_sys:and_inputs},
                                  targets={and_out_sys:and_targets},
                                  num_trials=(eps * and_inputs.shape[0] + 1))
        end = timeit.default_timer()
        sys_time = end - start

        # LOG TIMES, SPEEDUP PROVIDED BY COMPOSITION OVER SYSTEM

        msg = 'Training XOR model as AutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
        logger.info(msg)

        msg = 'Training XOR model as System for {0} epochs took {1} seconds'.format(eps, sys_time)
        print(msg)
        print("\n")
        logger.info(msg)

        speedup = np.round((sys_time / comp_time), decimals=2)
        msg = ('Training XOR model as AutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd'),
            (10, 'sgd'),
            (100, 'sgd')
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                    pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                    ])
    def test_xor_training_time(self, eps, opt,mode):

        # SET UP MECHANISMS FOR COMPOSITION

        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        # SET UP MECHANISMS FOR SYSTEM

        xor_in_sys = TransferMechanism(name='xor_in_sys',
                                       default_variable=np.zeros(2))

        xor_hid_sys = TransferMechanism(name='xor_hid_sys',
                                        default_variable=np.zeros(10),
                                        function=Logistic())

        xor_out_sys = TransferMechanism(name='xor_out_sys',
                                        default_variable=np.zeros(1),
                                        function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map = MappingProjection(name='hid_map',
                                    matrix=np.random.rand(2,10),
                                    sender=xor_in,
                                    receiver=xor_hid)

        out_map = MappingProjection(name='out_map',
                                    matrix=np.random.rand(10,1),
                                    sender=xor_hid,
                                    receiver=xor_out)

        # SET UP PROJECTIONS FOR SYSTEM

        hid_map_sys = MappingProjection(name='hid_map_sys',
                                        matrix=hid_map.matrix.copy(),
                                        sender=xor_in_sys,
                                        receiver=xor_hid_sys)

        out_map_sys = MappingProjection(name='out_map_sys',
                                        matrix=out_map.matrix.copy(),
                                        sender=xor_hid_sys,
                                        receiver=xor_out_sys)

        # SET UP COMPOSITION

        xor = AutodiffComposition(param_init_from_pnl=True,bin_execute=mode)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        # SET UP INPUTS AND TARGETS

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        # TIME TRAINING FOR COMPOSITION

        start = timeit.default_timer()
        result = xor.run(inputs={xor_in:xor_inputs},
                         targets={xor_out:xor_targets},
                         epochs=eps,
                         learning_rate=0.1,
                         controller=opt,
                         bin_execute=mode)
        end = timeit.default_timer()
        comp_time = end - start

        # SET UP SYSTEM

        xor_process = Process(pathway=[xor_in_sys,
                                       hid_map_sys,
                                       xor_hid_sys,
                                       out_map_sys,
                                       xor_out_sys],
                              learning=pnl.LEARNING)

        xor_sys = System(processes=[xor_process],
                         learning_rate=0.1)

        # TIME TRAINING FOR SYSTEM

        start = timeit.default_timer()
        results_sys = xor_sys.run(inputs={xor_in_sys:xor_inputs},
                                  targets={xor_out_sys:xor_targets},
                                  num_trials=(eps * xor_inputs.shape[0] + 1))
        end = timeit.default_timer()
        sys_time = end - start

        # LOG TIMES, SPEEDUP PROVIDED BY COMPOSITION OVER SYSTEM

        msg = 'Training XOR model as AutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
        logger.info(msg)

        msg = 'Training XOR model as System for {0} epochs took {1} seconds'.format(eps, sys_time)
        print(msg)
        print("\n")
        logger.info(msg)

        speedup = np.round((sys_time / comp_time), decimals=2)
        msg = ('Training XOR model as AutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd'),
            (10, 'sgd'),
            (100, 'sgd')
        ]
    )
    def test_semantic_net_training_time(self, eps, opt):

        # SET UP MECHANISMS FOR COMPOSITION:

        nouns_in = TransferMechanism(name="nouns_input",
                                     default_variable=np.zeros(8))

        rels_in = TransferMechanism(name="rels_input",
                                    default_variable=np.zeros(3))

        h1 = TransferMechanism(name="hidden_nouns",
                               default_variable=np.zeros(8),
                               function=Logistic())

        h2 = TransferMechanism(name="hidden_mixed",
                               default_variable=np.zeros(15),
                               function=Logistic())

        out_sig_I = TransferMechanism(name="sig_outs_I",
                                      default_variable=np.zeros(8),
                                      function=Logistic())

        out_sig_is = TransferMechanism(name="sig_outs_is",
                                       default_variable=np.zeros(12),
                                       function=Logistic())

        out_sig_has = TransferMechanism(name="sig_outs_has",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        out_sig_can = TransferMechanism(name="sig_outs_can",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        # SET UP MECHANISMS FOR SYSTEM

        nouns_in_sys = TransferMechanism(name="nouns_input_sys",
                                         default_variable=np.zeros(8))

        rels_in_sys = TransferMechanism(name="rels_input_sys",
                                        default_variable=np.zeros(3))

        h1_sys = TransferMechanism(name="hidden_nouns_sys",
                                   default_variable=np.zeros(8),
                                   function=Logistic())

        h2_sys = TransferMechanism(name="hidden_mixed_sys",
                                   default_variable=np.zeros(15),
                                   function=Logistic())

        out_sig_I_sys = TransferMechanism(name="sig_outs_I_sys",
                                          default_variable=np.zeros(8),
                                          function=Logistic())

        out_sig_is_sys = TransferMechanism(name="sig_outs_is_sys",
                                           default_variable=np.zeros(12),
                                           function=Logistic())

        out_sig_has_sys = TransferMechanism(name="sig_outs_has_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())

        out_sig_can_sys = TransferMechanism(name="sig_outs_can_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                         name="map_nouns_h1",
                                         sender=nouns_in,
                                         receiver=h1)

        map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                        name="map_rel_h2",
                                        sender=rels_in,
                                        receiver=h2)

        map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                                      name="map_h1_h2",
                                      sender=h1,
                                      receiver=h2)

        map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                                     name="map_h2_I",
                                     sender=h2,
                                     receiver=out_sig_I)

        map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                                      name="map_h2_is",
                                      sender=h2,
                                      receiver=out_sig_is)

        map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_has",
                                       sender=h2,
                                       receiver=out_sig_has)

        map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                                       name="map_h2_can",
                                       sender=h2,
                                       receiver=out_sig_can)

        # SET UP PROJECTIONS FOR SYSTEM

        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)

        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)

        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)

        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)

        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)

        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)

        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)

        # COMPOSITION FOR SEMANTIC NET

        sem_net = AutodiffComposition(param_init_from_pnl=True)

        sem_net.add_node(nouns_in)
        sem_net.add_node(rels_in)
        sem_net.add_node(h1)
        sem_net.add_node(h2)
        sem_net.add_node(out_sig_I)
        sem_net.add_node(out_sig_is)
        sem_net.add_node(out_sig_has)
        sem_net.add_node(out_sig_can)

        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)

        # INPUTS & OUTPUTS FOR SEMANTIC NET:

        nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
        relations = ['is', 'has', 'can']
        is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
                   'yellow']
        has_list = ['roots', 'leaves', 'bark', 'branches', 'skin', 'feathers', 'wings', 'gills', 'scales']
        can_list = ['grow', 'move', 'swim', 'fly', 'breathe', 'breathe underwater', 'breathe air', 'walk', 'photosynthesize']

        nouns_input = np.identity(len(nouns))

        rels_input = np.identity(len(relations))

        truth_nouns = np.identity(len(nouns))

        truth_is = np.zeros((len(nouns), len(is_list)))

        truth_is[0, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[1, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[2, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[3, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[4, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[5, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[6, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        truth_is[7, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

        truth_has = np.zeros((len(nouns), len(has_list)))

        truth_has[0, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[1, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[2, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[3, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[4, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        truth_has[7, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]

        truth_can = np.zeros((len(nouns), len(can_list)))

        truth_can[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[3, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[4, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[5, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[6, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        truth_can[7, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]

        # SETTING UP DICTIONARIES OF INPUTS/OUTPUTS FOR SEMANTIC NET

        inputs_dict = {}
        inputs_dict[nouns_in] = []
        inputs_dict[rels_in] = []

        targets_dict = {}
        targets_dict[out_sig_I] = []
        targets_dict[out_sig_is] = []
        targets_dict[out_sig_has] = []
        targets_dict[out_sig_can] = []

        for i in range(len(nouns)):
            for j in range(len(relations)):
                inputs_dict[nouns_in].append(nouns_input[i])
                inputs_dict[rels_in].append(rels_input[j])
                targets_dict[out_sig_I].append(truth_nouns[i])
                targets_dict[out_sig_is].append(truth_is[i])
                targets_dict[out_sig_has].append(truth_has[i])
                targets_dict[out_sig_can].append(truth_can[i])

        inputs_dict_sys = {}
        inputs_dict_sys[nouns_in_sys] = inputs_dict[nouns_in]
        inputs_dict_sys[rels_in_sys] = inputs_dict[rels_in]

        targets_dict_sys = {}
        targets_dict_sys[out_sig_I_sys] = targets_dict[out_sig_I]
        targets_dict_sys[out_sig_is_sys] = targets_dict[out_sig_is]
        targets_dict_sys[out_sig_has_sys] = targets_dict[out_sig_has]
        targets_dict_sys[out_sig_can_sys] = targets_dict[out_sig_can]

        # TIME TRAINING FOR COMPOSITION

        start = timeit.default_timer()
        result = sem_net.run(inputs=inputs_dict,
                             targets=targets_dict,
                             epochs=eps,
                             learning_rate=0.1,
                             controller=opt)
        end = timeit.default_timer()
        comp_time = end - start

        # SET UP SYSTEM

        p11 = Process(pathway=[nouns_in_sys,
                               map_nouns_h1_sys,
                               h1_sys,
                               map_h1_h2_sys,
                               h2_sys,
                               map_h2_I_sys,
                               out_sig_I_sys],
                      learning=pnl.LEARNING)

        p12 = Process(pathway=[rels_in_sys,
                               map_rels_h2_sys,
                               h2_sys,
                               map_h2_is_sys,
                               out_sig_is_sys],
                      learning=pnl.LEARNING)

        p21 = Process(pathway=[h2_sys,
                               map_h2_has_sys,
                               out_sig_has_sys],
                      learning=pnl.LEARNING)

        p22 = Process(pathway=[h2_sys,
                               map_h2_can_sys,
                               out_sig_can_sys],
                      learning=pnl.LEARNING)

        sem_net_sys = System(processes=[p11,
                                        p12,
                                        p21,
                                        p22,
                                        ],
                             learning_rate=0.1)

        # TIME TRAINING FOR SYSTEM

        start = timeit.default_timer()
        results = sem_net_sys.run(inputs=inputs_dict_sys,
                                  targets=targets_dict_sys,
                                  num_trials=(len(inputs_dict[nouns_in]) * eps + 1))
        end = timeit.default_timer()
        sys_time = end - start

        # LOG TIMES, SPEEDUP PROVIDED BY COMPOSITION OVER SYSTEM

        msg = 'Training Semantic net as AutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
        logger.info(msg)

        msg = 'Training Semantic net as System for {0} epochs took {1} seconds'.format(eps, sys_time)
        print(msg)
        print("\n")
        logger.info(msg)

        speedup = np.round((sys_time / comp_time), decimals=2)
        msg = ('Training Semantic net as AutodiffComposition for {0} epochs was {1} times faster than '
               'training it as System for {0} epochs.'.format(eps, speedup))
        print(msg)
        logger.info(msg)


@pytest.mark.pytorch
@pytest.mark.acidenticalness
class TestTrainingIdenticalness():

    @pytest.mark.parametrize(
        'eps, opt', [
            # (1, 'sgd'),
            (10, 'sgd'),
            # (40, 'sgd')
        ]
    )
    def test_semantic_net_training_identicalness(self, eps, opt):
        # SET UP MECHANISMS FOR SEMANTIC NET:

        nouns_in = TransferMechanism(name="nouns_input",
                                     default_variable=np.zeros(8))

        rels_in = TransferMechanism(name="rels_input",
                                    default_variable=np.zeros(3))

        h1 = TransferMechanism(name="hidden_nouns",
                               default_variable=np.zeros(8),
                               function=Logistic())

        h2 = TransferMechanism(name="hidden_mixed",
                               default_variable=np.zeros(15),
                               function=Logistic())

        out_sig_I = TransferMechanism(name="sig_outs_I",
                                      default_variable=np.zeros(8),
                                      function=Logistic())

        out_sig_is = TransferMechanism(name="sig_outs_is",
                                       default_variable=np.zeros(12),
                                       function=Logistic())

        out_sig_has = TransferMechanism(name="sig_outs_has",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        out_sig_can = TransferMechanism(name="sig_outs_can",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        # SET UP MECHANISMS FOR SYSTEM

        nouns_in_sys = TransferMechanism(name="nouns_input_sys",
                                         default_variable=np.zeros(8))

        rels_in_sys = TransferMechanism(name="rels_input_sys",
                                        default_variable=np.zeros(3))

        h1_sys = TransferMechanism(name="hidden_nouns_sys",
                                   default_variable=np.zeros(8),
                                   function=Logistic())

        h2_sys = TransferMechanism(name="hidden_mixed_sys",
                                   default_variable=np.zeros(15),
                                   function=Logistic())

        out_sig_I_sys = TransferMechanism(name="sig_outs_I_sys",
                                          default_variable=np.zeros(8),
                                          function=Logistic())

        out_sig_is_sys = TransferMechanism(name="sig_outs_is_sys",
                                           default_variable=np.zeros(12),
                                           function=Logistic())

        out_sig_has_sys = TransferMechanism(name="sig_outs_has_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())

        out_sig_can_sys = TransferMechanism(name="sig_outs_can_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())

        # SET UP PROJECTIONS FOR SEMANTIC NET

        map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                 name="map_nouns_h1",
                                 sender=nouns_in,
                                 receiver=h1)

        map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                    name="map_relh2",
                                    sender=rels_in,
                                    receiver=h2)

        map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                                    name="map_h1_h2",
                                    sender=h1,
                                    receiver=h2)

        map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                                    name="map_h2_I",
                                    sender=h2,
                                    receiver=out_sig_I)

        map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                                    name="map_h2_is",
                                    sender=h2,
                                    receiver=out_sig_is)

        map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                                    name="map_h2_has",
                                    sender=h2,
                                    receiver=out_sig_has)

        map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                                    name="map_h2_can",
                                    sender=h2,
                                    receiver=out_sig_can)

        # SET UP PROJECTIONS FOR SYSTEM

        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)

        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)

        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)

        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)

        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)

        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)

        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)

        # SET UP COMPOSITION FOR SEMANTIC NET
        sem_net = AutodiffComposition(param_init_from_pnl=True,
                                      learning_rate=0.5,
                                      optimizer_type=opt,
                                      )

        sem_net.add_node(nouns_in)
        sem_net.add_node(rels_in)
        sem_net.add_node(h1)
        sem_net.add_node(h2)
        sem_net.add_node(out_sig_I)
        sem_net.add_node(out_sig_is)
        sem_net.add_node(out_sig_has)
        sem_net.add_node(out_sig_can)

        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)
        # INPUTS & OUTPUTS FOR SEMANTIC NET:

        nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
        relations = ['is', 'has', 'can']
        is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
                   'yellow']
        has_list = ['roots', 'leaves', 'bark', 'branches', 'skin', 'feathers', 'wings', 'gills', 'scales']
        can_list = ['grow', 'move', 'swim', 'fly', 'breathe', 'breathe underwater', 'breathe air', 'walk', 'photosynthesize']

        nouns_input = np.identity(len(nouns))

        rels_input = np.identity(len(relations))

        truth_nouns = np.identity(len(nouns))

        truth_is = np.zeros((len(nouns), len(is_list)))

        truth_is[0, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[1, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[2, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[3, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[4, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[5, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[6, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        truth_is[7, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

        truth_has = np.zeros((len(nouns), len(has_list)))

        truth_has[0, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[1, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[2, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[3, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[4, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        truth_has[7, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]

        truth_can = np.zeros((len(nouns), len(can_list)))

        truth_can[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[3, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[4, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[5, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[6, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        truth_can[7, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]

        # SETTING UP DICTIONARY OF INPUTS/OUTPUTS FOR SEMANTIC NET

        inputs_dict = {}
        inputs_dict[nouns_in] = []
        inputs_dict[rels_in] = []

        targets_dict = {}
        targets_dict[out_sig_I] = []
        targets_dict[out_sig_is] = []
        targets_dict[out_sig_has] = []
        targets_dict[out_sig_can] = []

        for i in range(len(nouns)):
            for j in range(len(relations)):
                inputs_dict[nouns_in].append(nouns_input[i])
                inputs_dict[rels_in].append(rels_input[j])
                targets_dict[out_sig_I].append(truth_nouns[i])
                targets_dict[out_sig_is].append(truth_is[i])
                targets_dict[out_sig_has].append(truth_has[i])
                targets_dict[out_sig_can].append(truth_can[i])

        inputs_dict_sys = {}
        inputs_dict_sys[nouns_in_sys] = inputs_dict[nouns_in]
        inputs_dict_sys[rels_in_sys] = inputs_dict[rels_in]

        result = sem_net.run(inputs=inputs_dict)

        # comp_weights = sem_net.get_parameters()[0]

        # TRAIN COMPOSITION
        def g_f():
            yield {"inputs": inputs_dict,
                   "targets": targets_dict,
                   "epochs": eps}
        g = g_f()
        result = sem_net.learn(inputs=g_f)

        comp_weights = sem_net.get_parameters()

        # SET UP SYSTEM
        sem_net_sys = Composition()

        learning_components = sem_net_sys.add_backpropagation_learning_pathway(
            pathway=[
                nouns_in_sys,
                map_nouns_h1_sys,
                h1_sys,
                map_h1_h2_sys,
                h2_sys,
                map_h2_I_sys,
                out_sig_I_sys
            ],
            learning_rate=0.5
        )
        inputs_dict_sys[learning_components[pnl.TARGET_MECHANISM]] = targets_dict[out_sig_I]

        learning_components = sem_net_sys.add_backpropagation_learning_pathway(
            pathway=[
                rels_in_sys,
                map_rels_h2_sys,
                h2_sys,
                map_h2_is_sys,
                out_sig_is_sys
            ],
            learning_rate=0.5
        )
        inputs_dict_sys[learning_components[pnl.TARGET_MECHANISM]] = targets_dict[out_sig_is]

        learning_components = sem_net_sys.add_backpropagation_learning_pathway(
            pathway=[
                h2_sys,
                map_h2_has_sys,
                out_sig_has_sys
            ],
            learning_rate=0.5
        )
        inputs_dict_sys[learning_components[pnl.TARGET_MECHANISM]] = targets_dict[out_sig_has]

        learning_components = sem_net_sys.add_backpropagation_learning_pathway(
            pathway=[
                h2_sys,
                map_h2_can_sys,
                out_sig_can_sys
            ],
            learning_rate=0.5
        )
        inputs_dict_sys[learning_components[pnl.TARGET_MECHANISM]] = targets_dict[out_sig_can]

        # TRAIN SYSTEM
        results = sem_net_sys.learn(inputs=inputs_dict_sys,
                                  num_trials=(len(inputs_dict_sys[nouns_in_sys]) * eps))

        # CHECK THAT PARAMETERS FOR COMPOSITION, SYSTEM ARE SAME

        assert np.allclose(comp_weights[map_nouns_h1], map_nouns_h1_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_rels_h2], map_rels_h2_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h1_h2], map_h1_h2_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_I], map_h2_I_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_is], map_h2_is_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_has], map_h2_has_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_can], map_h2_can_sys.get_mod_matrix(sem_net_sys))

    def test_identicalness_of_input_types(self):
        # SET UP MECHANISMS FOR COMPOSITION
        from copy import copy
        hid_map_mat = np.random.rand(2, 10)
        out_map_mat = np.random.rand(10, 1)
        xor_in_dict = TransferMechanism(name='xor_in',
                                        default_variable=np.zeros(2))

        xor_hid_dict = TransferMechanism(name='xor_hid',
                                         default_variable=np.zeros(10),
                                         function=Logistic())

        xor_out_dict = TransferMechanism(name='xor_out',
                                         default_variable=np.zeros(1),
                                         function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map_dict = MappingProjection(name='hid_map',
                                         matrix=copy(hid_map_mat),
                                         sender=xor_in_dict,
                                         receiver=xor_hid_dict)

        out_map_dict = MappingProjection(name='out_map',
                                         matrix=copy(out_map_mat),
                                         sender=xor_hid_dict,
                                         receiver=xor_out_dict)

        # SET UP COMPOSITION

        xor_dict = AutodiffComposition(param_init_from_pnl=True)

        xor_dict.add_node(xor_in_dict)
        xor_dict.add_node(xor_hid_dict)
        xor_dict.add_node(xor_out_dict)

        xor_dict.add_projection(sender=xor_in_dict, projection=hid_map_dict, receiver=xor_hid_dict)
        xor_dict.add_projection(sender=xor_hid_dict, projection=out_map_dict, receiver=xor_out_dict)
        # SET UP INPUTS AND TARGETS

        xor_inputs_dict = np.array(  # the inputs we will provide to the model
                [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

        xor_targets_dict = np.array(  # the outputs we wish to see from the model
                [[0],
                 [1],
                 [1],
                 [0]])

        input_dict = {
                "inputs": {
                    xor_in_dict: xor_inputs_dict
                },
                "targets": {
                    xor_out_dict: xor_targets_dict
                }
            }

        result_dict = xor_dict.learn(inputs=input_dict)

        # SET UP MECHANISMS FOR COMPOSITION
        xor_in_func = TransferMechanism(name='xor_in',
                                        default_variable=np.zeros(2))

        xor_hid_func = TransferMechanism(name='xor_hid',
                                         default_variable=np.zeros(10),
                                         function=Logistic())

        xor_out_func = TransferMechanism(name='xor_out',
                                         default_variable=np.zeros(1),
                                         function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map_func = MappingProjection(name='hid_map',
                                         matrix=copy(hid_map_mat),
                                         sender=xor_in_func,
                                         receiver=xor_hid_func)

        out_map_func = MappingProjection(name='out_map',
                                         matrix=copy(out_map_mat),
                                         sender=xor_hid_func,
                                         receiver=xor_out_func)

        # SET UP COMPOSITION

        xor_func = AutodiffComposition(param_init_from_pnl=True)

        xor_func.add_node(xor_in_func)
        xor_func.add_node(xor_hid_func)
        xor_func.add_node(xor_out_func)

        xor_func.add_projection(sender=xor_in_func, projection=hid_map_func, receiver=xor_hid_func)
        xor_func.add_projection(sender=xor_hid_func, projection=out_map_func, receiver=xor_out_func)

        # SET UP INPUTS AND TARGETS

        xor_inputs_func = np.array(  # the inputs we will provide to the model
                [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

        xor_targets_func = np.array(  # the outputs we wish to see from the model
                [[0],
                 [1],
                 [1],
                 [0]])

        def get_inputs():
            return {
                "inputs": {
                    xor_in_func: xor_inputs_func
                },
                "targets": {
                    xor_out_func: xor_targets_func
                }
            }

        result_func = xor_func.learn(inputs=get_inputs)

        # SET UP MECHANISMS FOR COMPOSITION
        xor_in_gen = TransferMechanism(name='xor_in',
                                       default_variable=np.zeros(2))

        xor_hid_gen = TransferMechanism(name='xor_hid',
                                        default_variable=np.zeros(10),
                                        function=Logistic())

        xor_out_gen = TransferMechanism(name='xor_out',
                                        default_variable=np.zeros(1),
                                        function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map_gen = MappingProjection(name='hid_map',
                                        matrix=copy(hid_map_mat),
                                        sender=xor_in_gen,
                                        receiver=xor_hid_gen)

        out_map_gen = MappingProjection(name='out_map',
                                        matrix=copy(out_map_mat),
                                        sender=xor_hid_gen,
                                        receiver=xor_out_gen)

        # SET UP COMPOSITION

        xor_gen = AutodiffComposition(param_init_from_pnl=True)

        xor_gen.add_node(xor_in_gen)
        xor_gen.add_node(xor_hid_gen)
        xor_gen.add_node(xor_out_gen)

        xor_gen.add_projection(sender=xor_in_gen, projection=hid_map_gen, receiver=xor_hid_gen)
        xor_gen.add_projection(sender=xor_hid_gen, projection=out_map_gen, receiver=xor_out_gen)
        # SET UP INPUTS AND TARGETS

        xor_inputs_gen = np.array(  # the inputs we will provide to the model
                [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

        xor_targets_gen = np.array(  # the outputs we wish to see from the model
                [[0],
                 [1],
                 [1],
                 [0]])

        def get_inputs_gen():
            yield {
                "inputs": {
                    xor_in_gen: xor_inputs_gen
                },
                "targets": {
                    xor_out_gen: xor_targets_gen
                }
            }

        g = get_inputs_gen()

        result_gen = xor_gen.learn(inputs=g)

        # SET UP MECHANISMS FOR COMPOSITION
        xor_in_gen_func = TransferMechanism(name='xor_in',
                                            default_variable=np.zeros(2))

        xor_hid_gen_func = TransferMechanism(name='xor_hid',
                                             default_variable=np.zeros(10),
                                             function=Logistic())

        xor_out_gen_func = TransferMechanism(name='xor_out',
                                             default_variable=np.zeros(1),
                                             function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map_gen_func = MappingProjection(name='hid_map',
                                             matrix=copy(hid_map_mat),
                                             sender=xor_in_gen_func,
                                             receiver=xor_hid_gen_func)

        out_map_gen_func = MappingProjection(name='out_map',
                                             matrix=copy(out_map_mat),
                                             sender=xor_hid_gen_func,
                                             receiver=xor_out_gen_func)

        # SET UP COMPOSITION

        xor_gen_func = AutodiffComposition(param_init_from_pnl=True)

        xor_gen_func.add_node(xor_in_gen_func)
        xor_gen_func.add_node(xor_hid_gen_func)
        xor_gen_func.add_node(xor_out_gen_func)

        xor_gen_func.add_projection(sender=xor_in_gen_func, projection=hid_map_gen_func, receiver=xor_hid_gen_func)
        xor_gen_func.add_projection(sender=xor_hid_gen_func, projection=out_map_gen_func, receiver=xor_out_gen_func)
        # SET UP INPUTS AND TARGETS

        xor_inputs_gen_func = np.array(  # the inputs we will provide to the model
                [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

        xor_targets_gen_func = np.array(  # the outputs we wish to see from the model
                [[0],
                 [1],
                 [1],
                 [0]])

        def get_inputs_gen_func():
            yield {
                "inputs": {
                    xor_in_gen_func: xor_inputs_gen_func
                },
                "targets": {
                    xor_out_gen_func: xor_targets_gen_func
                }
            }

        result_gen_func = xor_gen_func.learn(inputs=get_inputs_gen_func)

        assert result_dict == result_func == result_gen == result_gen_func


@pytest.mark.pytorch
@pytest.mark.aclogging
class TestACLogging:
    def test_autodiff_logging(self):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection()
        out_map = MappingProjection()

        xor = AutodiffComposition(param_init_from_pnl=True)
        xor.add_backpropagation_learning_pathway([xor_in, hid_map, xor_hid, out_map, xor_out])
        hid_map.set_log_conditions('matrix', pnl.LogCondition.TRIAL)
        out_map.set_log_conditions('matrix', pnl.LogCondition.TRIAL)

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        # train model for a few epochs
        num_epochs = 10
        xor.learn(inputs={"inputs": {xor_in: xor_inputs},
                        "targets": {xor_out: xor_targets},
                        "epochs": num_epochs})

        exec_id = xor.default_execution_id

        in_np_dict_vals = xor_in.log.nparray_dictionary()[exec_id]['value']
        in_np_vals = xor_in.log.nparray()[1][1][4][1:]

        hid_map_np_dict_mats = hid_map.log.nparray_dictionary()[exec_id]['matrix']
        hid_map_np_mats = np.array(hid_map.log.nparray()[1][1][5][1:])

        hid_np_dict_vals = xor_hid.log.nparray_dictionary()[exec_id]['value']

        out_map_np_dict_mats = out_map.log.nparray_dictionary()[exec_id]['matrix']
        out_map_np_mats = np.array(out_map.log.nparray()[1][1][5][1:])

        out_np_dict_vals = xor_out.log.nparray_dictionary()[exec_id]['value']

        expected_length = len(xor_inputs) * num_epochs

        assert np.all(in_np_dict_vals[0:4] == xor_inputs)
        assert np.all(np.array(in_np_vals) == in_np_dict_vals)
        assert in_np_dict_vals.shape == (expected_length, xor_in.size)

        assert hid_map_np_dict_mats.shape == (expected_length, xor_in.size, xor_hid.size)
        assert hid_map_np_mats.shape == hid_map_np_dict_mats.shape
        assert np.all(hid_map_np_mats[3] == hid_map_np_dict_mats[3])  # CW: 3 is arbitrary. you can use any index

        assert hid_np_dict_vals.shape == (expected_length, xor_hid.size)

        assert out_map_np_dict_mats.shape == (expected_length, xor_hid.size, xor_out.size)
        assert out_map_np_mats.shape == out_map_np_dict_mats.shape
        assert np.all(out_map_np_mats[3] == out_map_np_dict_mats[3])

        assert out_np_dict_vals.shape == (expected_length, xor_out.size)

        xor_out.log.print_entries()

@pytest.mark.pytorch
@pytest.mark.acnested
class TestNested:

    @pytest.mark.parametrize(
        'num_epochs, learning_rate, patience, min_delta', [
            (400, 4, 10, .00001),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
                                     ])
    def test_xor_nested_train_then_no_train(self, num_epochs, learning_rate,
                                            patience, min_delta, mode):
        # the inputs we will provide to the model
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # the outputs we wish to see from the model
        xor_targets = np.array([[0], [1], [1], [0]])

        # -----------------------------------------------------------------

        xor_in = pnl.TransferMechanism(name='xor_in',
                                       default_variable=np.zeros(2))

        xor_hid = pnl.TransferMechanism(name='xor_hid',
                                        default_variable=np.zeros(10),
                                        function=pnl.core.components.functions.transferfunctions.Logistic())

        xor_out = pnl.TransferMechanism(name='xor_out',
                                        default_variable=np.zeros(1),
                                        function=pnl.core.components.functions.transferfunctions.Logistic())

        hid_map = pnl.MappingProjection(name='input_to_hidden',
                                        matrix=np.random.randn(2, 10) * 0.1,
                                        sender=xor_in,
                                        receiver=xor_hid)

        out_map = pnl.MappingProjection(name='hidden_to_output',
                                        matrix=np.random.randn(10, 1) * 0.1,
                                        sender=xor_hid,
                                        receiver=xor_out)

        # -----------------------------------------------------------------

        xor_autodiff = AutodiffComposition(
            param_init_from_pnl=True,
            learning_rate=learning_rate,
        )
        xor_autodiff.add_backpropagation_learning_pathway([xor_in, hid_map, xor_hid, out_map, xor_out])

        # -----------------------------------------------------------------

        no_training_input_dict = {xor_in: xor_inputs}
        input_dict = {'inputs': {xor_in: xor_inputs }, 'targets': {xor_out: xor_targets}, 'epochs': num_epochs}

        parentComposition = pnl.Composition()
        parentComposition.add_node(xor_autodiff)
        
        input = {xor_autodiff: input_dict}
        no_training_input = {xor_autodiff: no_training_input_dict}
        
        learning_context = Context()
        result1 = xor_autodiff.learn(inputs=input_dict, bin_execute=mode, epochs=num_epochs, context=learning_context, patience=patience, min_delta=min_delta)
        result1 = np.array(result1).flatten()
        assert np.allclose(result1, np.array(xor_targets).flatten(), atol=0.1)
        result2 = parentComposition.run(inputs=no_training_input, bin_execute=mode, context=learning_context)

        assert np.allclose(result2, [[0]], atol=0.1)

    @pytest.mark.parametrize(
        'num_epochs, learning_rate, patience, min_delta', [
            (400, 4, 10, .00001),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                    #   pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
                                     ])
    def test_xor_nested_no_train_then_train(self, num_epochs, learning_rate,
                                            patience, min_delta, mode):
        # the inputs we will provide to the model
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # the outputs we wish to see from the model
        xor_targets = np.array([[0], [1], [1], [0]])

        # -----------------------------------------------------------------

        xor_in = pnl.TransferMechanism(name='xor_in',
                                       default_variable=np.zeros(2))

        xor_hid = pnl.TransferMechanism(name='xor_hid',
                                        default_variable=np.zeros(10),
                                        function=pnl.core.components.functions.transferfunctions.Logistic())

        xor_out = pnl.TransferMechanism(name='xor_out',
                                        default_variable=np.zeros(1),
                                        function=pnl.core.components.functions.transferfunctions.Logistic())

        hid_map = pnl.MappingProjection(name='input_to_hidden',
                                        matrix=np.random.randn(2, 10) * 0.1,
                                        sender=xor_in,
                                        receiver=xor_hid)

        out_map = pnl.MappingProjection(name='hidden_to_output',
                                        matrix=np.random.randn(10, 1) * 0.1,
                                        sender=xor_hid,
                                        receiver=xor_out)

        # -----------------------------------------------------------------

        xor_autodiff = AutodiffComposition(
            param_init_from_pnl=True,
            learning_rate=learning_rate,
        )

        xor_autodiff.add_backpropagation_learning_pathway([xor_in, hid_map, xor_hid, out_map, xor_out])

        # -----------------------------------------------------------------

        no_training_input_dict = {xor_in: xor_inputs}
        input_dict = {'inputs': {xor_in: xor_inputs}, 'targets': {xor_out: xor_targets}, 'epochs': num_epochs}

        parentComposition = pnl.Composition()
        parentComposition.add_node(xor_autodiff)
        input = {xor_autodiff: input_dict}
        no_training_input = {xor_autodiff: no_training_input_dict}
        learning_context = Context()
        result1 = xor_autodiff.run(inputs=input[xor_autodiff]['inputs'], bin_execute=mode, context=learning_context)
        xor_autodiff.learn(inputs=input_dict, bin_execute=mode, context=learning_context, patience=patience, min_delta=min_delta)
        result2 = parentComposition.run(inputs=no_training_input, bin_execute=mode, context=learning_context)

        assert np.allclose(result2, [[0]], atol=0.1)

    # CW 12/21/18: Test is failing due to bugs, will fix later
    # @pytest.mark.parametrize(
    #     'num_epochs, learning_rate, patience, min_delta', [
    #         (2000, 4, 10, .00001),
    #     ]
    # )
    # def test_xor_nest_not_origin_after_train(self, num_epochs, learning_rate, patience, min_delta):
    #     xor_inputs = np.array(  # the inputs we will provide to the model
    #         [[0, 0],
    #          [0, 1],
    #          [1, 0],
    #          [1, 1]])
    #
    #     xor_targets = np.array(  # the outputs we wish to see from the model
    #         [[0],
    #          [1],
    #          [1],
    #          [0]])
    #
    #     # -----------------------------------------------------------------
    #
    #     xor_in = pnl.TransferMechanism(name='xor_in',
    #                                    default_variable=np.zeros(2))
    #
    #     xor_hid = pnl.TransferMechanism(name='xor_hid',
    #                                     default_variable=np.zeros(10),
    #                                     function=pnl.core.components.functions.transferfunctions.Logistic())
    #
    #     xor_out = pnl.TransferMechanism(name='xor_out',
    #                                     default_variable=np.zeros(1),
    #                                     function=pnl.core.components.functions.transferfunctions.Logistic())
    #
    #     hid_map = pnl.MappingProjection(name='input_to_hidden',
    #                                     matrix=np.random.randn(2, 10) * 0.1,
    #                                     sender=xor_in,
    #                                     receiver=xor_hid)
    #
    #     out_map = pnl.MappingProjection(name='hidden_to_output',
    #                                     matrix=np.random.randn(10, 1) * 0.1,
    #                                     sender=xor_hid,
    #                                     receiver=xor_out)
    #
    #     # -----------------------------------------------------------------
    #
    #     xor_autodiff = AutodiffComposition(
    #         param_init_from_pnl=True,
    #         patience=patience,
    #         min_delta=min_delta,
    #         learning_rate=learning_rate,
    #         randomize=False,
    #         learning_enabled=True
    #     )
    #
    #     xor_autodiff.add_node(xor_in)
    #     xor_autodiff.add_node(xor_hid)
    #     xor_autodiff.add_node(xor_out)
    #
    #     xor_autodiff.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
    #     xor_autodiff.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
    #
    #     # -----------------------------------------------------------------
    #
    #     input_dict = {'inputs': {xor_in: xor_inputs}, 'targets': {xor_out: xor_targets}, 'epochs': num_epochs}
    #     xor_autodiff.run(inputs = input_dict)
    #     myTransfer = pnl.TransferMechanism(size = 2)
    #     myMappingProj = pnl.MappingProjection(sender = myTransfer, receiver = xor_autodiff)
    #
    #     no_training_input_dict = {xor_in: xor_inputs}
    #
    #     parentComposition = pnl.Composition()
    #     parentComposition.add_node(myTransfer)
    #     parentComposition.add_node(xor_autodiff)
    #     parentComposition.add_projection(myMappingProj, sender=myTransfer, receiver=xor_autodiff)
    #     xor_autodiff.learning_enabled = False
    #
    #     no_training_input = {myTransfer: no_training_input_dict}
    #
    #     result = parentComposition.run(inputs=no_training_input)
    #
    #     assert np.allclose(result, [[0]], atol=0.1)

    @pytest.mark.parametrize(
        'eps, opt', [
            (1, 'sgd'),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
                                     ])
    def test_semantic_net_nested(self, eps, opt, mode):

        # SET UP MECHANISMS FOR SEMANTIC NET:

        nouns_in = TransferMechanism(name="nouns_input",
                                     default_variable=np.zeros(8))

        rels_in = TransferMechanism(name="rels_input",
                                    default_variable=np.zeros(3))

        h1 = TransferMechanism(name="hidden_nouns",
                               default_variable=np.zeros(8),
                               function=Logistic())

        h2 = TransferMechanism(name="hidden_mixed",
                               default_variable=np.zeros(15),
                               function=Logistic())

        out_sig_I = TransferMechanism(name="sig_outs_I",
                                      default_variable=np.zeros(8),
                                      function=Logistic())

        out_sig_is = TransferMechanism(name="sig_outs_is",
                                       default_variable=np.zeros(12),
                                       function=Logistic())

        out_sig_has = TransferMechanism(name="sig_outs_has",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        out_sig_can = TransferMechanism(name="sig_outs_can",
                                        default_variable=np.zeros(9),
                                        function=Logistic())

        # SET UP MECHANISMS FOR SYSTEM

        nouns_in_sys = TransferMechanism(name="nouns_input_sys",
                                         default_variable=np.zeros(8))

        rels_in_sys = TransferMechanism(name="rels_input_sys",
                                        default_variable=np.zeros(3))

        h1_sys = TransferMechanism(name="hidden_nouns_sys",
                                   default_variable=np.zeros(8),
                                   function=Logistic())

        h2_sys = TransferMechanism(name="hidden_mixed_sys",
                                   default_variable=np.zeros(15),
                                   function=Logistic())

        out_sig_I_sys = TransferMechanism(name="sig_outs_I_sys",
                                          default_variable=np.zeros(8),
                                          function=Logistic())

        out_sig_is_sys = TransferMechanism(name="sig_outs_is_sys",
                                           default_variable=np.zeros(12),
                                           function=Logistic())

        out_sig_has_sys = TransferMechanism(name="sig_outs_has_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())

        out_sig_can_sys = TransferMechanism(name="sig_outs_can_sys",
                                            default_variable=np.zeros(9),
                                            function=Logistic())

        # SET UP PROJECTIONS FOR SEMANTIC NET

        map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                 name="map_nouns_h1",
                                 sender=nouns_in,
                                 receiver=h1)

        map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                    name="map_relh2",
                                    sender=rels_in,
                                    receiver=h2)

        map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                                    name="map_h1_h2",
                                    sender=h1,
                                    receiver=h2)

        map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                                    name="map_h2_I",
                                    sender=h2,
                                    receiver=out_sig_I)

        map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                                    name="map_h2_is",
                                    sender=h2,
                                    receiver=out_sig_is)

        map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                                    name="map_h2_has",
                                    sender=h2,
                                    receiver=out_sig_has)

        map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                                    name="map_h2_can",
                                    sender=h2,
                                    receiver=out_sig_can)

        # SET UP PROJECTIONS FOR SYSTEM

        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)

        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)

        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)

        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)

        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)

        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)

        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)

        # SET UP COMPOSITION FOR SEMANTIC NET

        sem_net = AutodiffComposition(param_init_from_pnl=True,
                                      learning_rate=0.5,
                                      optimizer_type=opt)

        sem_net.add_node(nouns_in)
        sem_net.add_node(rels_in)
        sem_net.add_node(h1)
        sem_net.add_node(h2)
        sem_net.add_node(out_sig_I)
        sem_net.add_node(out_sig_is)
        sem_net.add_node(out_sig_has)
        sem_net.add_node(out_sig_can)

        sem_net.add_projection(sender=nouns_in, projection=map_nouns_h1, receiver=h1)
        sem_net.add_projection(sender=rels_in, projection=map_rels_h2, receiver=h2)
        sem_net.add_projection(sender=h1, projection=map_h1_h2, receiver=h2)
        sem_net.add_projection(sender=h2, projection=map_h2_I, receiver=out_sig_I)
        sem_net.add_projection(sender=h2, projection=map_h2_is, receiver=out_sig_is)
        sem_net.add_projection(sender=h2, projection=map_h2_has, receiver=out_sig_has)
        sem_net.add_projection(sender=h2, projection=map_h2_can, receiver=out_sig_can)

        # INPUTS & OUTPUTS FOR SEMANTIC NET:

        nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
        relations = ['is', 'has', 'can']
        is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
                   'yellow']
        has_list = ['roots', 'leaves', 'bark', 'branches', 'skin', 'feathers', 'wings', 'gills', 'scales']
        can_list = ['grow', 'move', 'swim', 'fly', 'breathe', 'breathe underwater', 'breathe air', 'walk', 'photosynthesize']

        nouns_input = np.identity(len(nouns))

        rels_input = np.identity(len(relations))

        truth_nouns = np.identity(len(nouns))

        truth_is = np.zeros((len(nouns), len(is_list)))

        truth_is[0, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[1, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        truth_is[2, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[3, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        truth_is[4, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[5, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        truth_is[6, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        truth_is[7, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

        truth_has = np.zeros((len(nouns), len(has_list)))

        truth_has[0, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[1, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
        truth_has[2, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[3, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        truth_has[4, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
        truth_has[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        truth_has[7, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]

        truth_can = np.zeros((len(nouns), len(can_list)))

        truth_can[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[3, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
        truth_can[4, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[5, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        truth_can[6, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
        truth_can[7, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]

        # SETTING UP DICTIONARY OF INPUTS/OUTPUTS FOR SEMANTIC NET

        inputs_dict = {}
        inputs_dict[nouns_in] = []
        inputs_dict[rels_in] = []

        targets_dict = {}
        targets_dict[out_sig_I] = []
        targets_dict[out_sig_is] = []
        targets_dict[out_sig_has] = []
        targets_dict[out_sig_can] = []

        for i in range(len(nouns)):
            for j in range(len(relations)):
                inputs_dict[nouns_in].append(nouns_input[i])
                inputs_dict[rels_in].append(rels_input[j])
                targets_dict[out_sig_I].append(truth_nouns[i])
                targets_dict[out_sig_is].append(truth_is[i])
                targets_dict[out_sig_has].append(truth_has[i])
                targets_dict[out_sig_can].append(truth_can[i])

        # comp_weights = sem_net.get_parameters()[0]

        # TRAIN COMPOSITION
        input_dict = {"inputs": inputs_dict,
                      "targets": targets_dict,
                      "epochs": eps}

        parentComposition = pnl.Composition()
        parentComposition.add_node(sem_net)

        input = {sem_net: input_dict}
        no_training_input = {sem_net: inputs_dict.copy()}

        sem_net.learn(inputs=input_dict, bin_execute=mode)

        if mode != 'Python':
            #FIXME: Enable the rest of the test when recompilation is supported
            return

        parentComposition.run(inputs=no_training_input)

@pytest.mark.pytorch
class TestBatching:
    def test_call_before_minibatch(self):
        # SET UP MECHANISMS FOR COMPOSITION

        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map = MappingProjection(name='hid_map',
                                    matrix=np.random.rand(2, 10),
                                    sender=xor_in,
                                    receiver=xor_hid)

        out_map = MappingProjection(name='out_map',
                                    matrix=np.random.rand(10, 1),
                                    sender=xor_hid,
                                    receiver=xor_out)

        # SET UP COMPOSITION

        xor = AutodiffComposition(param_init_from_pnl=True,
                                  learning_rate=10)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        # SET UP INPUTS AND TARGETS

        xor_inputs_1 = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets_1 = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        # TRAIN COMPOSITION
        inputs_dict_1 = {"inputs": {xor_in: xor_inputs_1},
                         "targets": {xor_out: xor_targets_1},
                         "epochs": 1}

        a = [0]

        def cbm(a):
            a[0] += 1

        xor.learn(
            inputs=inputs_dict_1,
            call_before_minibatch=lambda: cbm(a)
        )

        assert a[0] == 4

    def test_call_after_minibatch(self):
        # SET UP MECHANISMS FOR COMPOSITION

        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map = MappingProjection(name='hid_map',
                                    matrix=np.random.rand(2, 10),
                                    sender=xor_in,
                                    receiver=xor_hid)

        out_map = MappingProjection(name='out_map',
                                    matrix=np.random.rand(10, 1),
                                    sender=xor_hid,
                                    receiver=xor_out)

        # SET UP COMPOSITION

        xor = AutodiffComposition(param_init_from_pnl=True,
                                  learning_rate=10)

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        # SET UP INPUTS AND TARGETS

        xor_inputs_1 = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets_1 = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        # TRAIN COMPOSITION
        inputs_dict_1 = {"inputs": {xor_in: xor_inputs_1},
                         "targets": {xor_out: xor_targets_1},
                         "epochs": 1}

        a = [0]

        def cam(a):
            a[0] += 1

        xor.learn(
            inputs=inputs_dict_1,
            call_after_minibatch=lambda: cam(a)
        )

        assert a[0] == 4

    @pytest.mark.parametrize(
        'eps', (1, 5, 10, 100)
    )
    def test_batching_with_epochs_specified(self, eps):
        # SET UP MECHANISMS FOR COMPOSITION

        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        # SET UP PROJECTIONS FOR COMPOSITION

        hid_map = MappingProjection(name='hid_map',
                                    matrix=np.random.rand(2, 10),
                                    sender=xor_in,
                                    receiver=xor_hid)

        out_map = MappingProjection(name='out_map',
                                    matrix=np.random.rand(10, 1),
                                    sender=xor_hid,
                                    receiver=xor_out)

        # SET UP COMPOSITION

        xor = AutodiffComposition(param_init_from_pnl=True,
                                  learning_rate=10,
                                  # optimizer_type=opt
                                  )

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)
        # SET UP INPUTS AND TARGETS

        xor_inputs_1 = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets_1 = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        c1 = Context(execution_id='context1')

        # TRAIN COMPOSITION
        inputs_dict_1 = {"inputs": {xor_in: xor_inputs_1},
                         "targets": {xor_out: xor_targets_1},
                         "epochs": eps}

        xor.learn(
            inputs=inputs_dict_1,
            context=c1,
            minibatch_size=2
        )

        c2 = Context(execution_id='context2')

        xor_inputs_2 = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1]])

        xor_targets_2 = np.array(  # the outputs we wish to see from the model
            [[0],
             [1]])

        inputs_dict_2 = {"inputs": {xor_in: xor_inputs_2},
                         "targets": {xor_out: xor_targets_2},
                         "epochs": 1}

        xor_inputs_3 = np.array(
            [[1, 0],
             [1, 1]]
        )

        xor_targets_3 = np.array(
            [[1],
             [0]]
        )

        inputs_dict_3 = {"inputs": {xor_in: xor_inputs_3},
                         "targets": {xor_out: xor_targets_3},
                         "epochs": 1}
        for _ in range(eps):
            xor.learn(
                inputs=inputs_dict_2,
                context=c2,
                minibatch_size=TRAINING_SET
            )
            xor.learn(
                inputs=inputs_dict_3,
                context=c2,
                minibatch_size=TRAINING_SET
            )

        c1_results = xor.parameters.results._get(c1)
        c2_results = xor.parameters.results._get(c2)

        assert np.allclose(c1_results[0][:2], c2_results[0][:2])
        assert np.allclose(c1_results[0][2:], c2_results[0][2:])

    def test_cross_entropy_loss(self):
        import torch

        m1 = pnl.TransferMechanism()
        p = pnl.MappingProjection()
        m2 = pnl.TransferMechanism()
        adc = pnl.AutodiffComposition(loss_spec='crossentropy')

        adc.add_linear_processing_pathway([m1, p, m2])
        adc._build_pytorch_representation()

        classes = torch.Tensor([2, 1])
        target = torch.Tensor([1])

        # Equation for loss taken from https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
        assert np.allclose(adc.loss(classes, target).detach().numpy(), -1 + np.log(np.exp(2) + np.exp(1)))
