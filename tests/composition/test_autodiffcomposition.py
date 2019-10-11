import logging
import timeit as timeit

import numpy as np

import pytest

import psyneulink as pnl

from psyneulink.core.components.functions.transferfunctions import Logistic
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
    #     assert comp.target_CIM_states == {}

    def test_pytorch_representation(self):
        comp = AutodiffComposition()
        assert comp.pytorch_representation == None

    def test_report_prefs(self):
        comp = AutodiffComposition()
        assert comp.input_CIM.reportOutputPref == False
        assert comp.output_CIM.reportOutputPref == False
        # assert comp.target_CIM.reportOutputPref == False

    def test_patience(self):
        comp = AutodiffComposition(patience=10)
        assert comp.patience == 10


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
    def test_training_then_processing(self,mode):
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
        results_before_proc = xor.run(inputs = {"inputs": {xor_in:xor_inputs},
                                                "targets": {xor_out:xor_targets},
                                               "epochs": 10},bin_execute = mode)

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

        xor.run(inputs = {"inputs": {xor_in:xor_inputs},
                          "targets": {xor_out:xor_targets},
                          "epochs": 10}, bin_execute = mode)

    @pytest.mark.parametrize("mode", ['Python',
                                    pytest.param('LLVMRun', marks=[pytest.mark.llvm,pytest.mark.skip]), # Not implemented?
                                    ])
    def test_pytorch_loss_spec(self,mode):
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
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        xor.run(inputs={"inputs": {xor_in:xor_inputs},
                          "targets": {xor_out:xor_targets},
                          "epochs": 10}, bin_execute = mode)
        xor.run(inputs={"inputs": {xor_in: xor_inputs},
                        "targets": {xor_out: xor_targets},
                        "epochs": 10}, bin_execute = mode)


    @pytest.mark.parametrize(
        'learning_rate, weight_decay, optimizer_type', [
            (10, 0, 'sgd'), (1.5, 1, 'sgd'),  (1.5, 1, 'adam'),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                    pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                    ])
    def test_optimizer_specs(self, learning_rate, weight_decay, optimizer_type,mode):
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
        results_before_proc = xor.run(inputs = {"inputs": {xor_in:xor_inputs},
                                                "targets": {xor_out:xor_targets},
                                               "epochs": 10}, bin_execute = mode)

    # test whether pytorch parameters and projections are kept separate (at diff. places in memory)
    @pytest.mark.parametrize("mode", ['Python',
                                    pytest.param('LLVMRun', marks=pytest.mark.llvm),
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
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        # train the model for a few epochs
        result = xor.run(inputs={"inputs": {xor_in:xor_inputs},
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
                                    ])
    def test_get_params(self,mode):

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
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

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
        weights_get_params = xor.get_parameters()[0]
        weights_straight_1 = xor.parameters.pytorch_representation.get(xor).params[0]
        weights_straight_2 = xor.parameters.pytorch_representation.get(xor).params[1]

        # check that parameter copies obtained from get_parameters are the same as the
        # projections and parameters from pytorch
        assert np.allclose(hid_map.parameters.matrix.get(None), weights_get_params[hid_map])
        assert np.allclose(weights_straight_1.detach().numpy(), weights_get_params[hid_map])
        assert np.allclose(out_map.parameters.matrix.get(None), weights_get_params[out_map])
        assert np.allclose(weights_straight_2.detach().numpy(), weights_get_params[out_map])

        # call run to train the pytorch parameters
        results = xor.run(inputs={"inputs": {xor_in:xor_inputs},
                                  "targets": {xor_out:xor_targets},
                                  "epochs": 10},bin_execute = mode)


        # check that the parameter copies obtained from get_parameters have not changed with the
        # pytorch parameters during training (and are thus at a different memory location)
        assert not np.allclose(weights_straight_1.detach().numpy(), weights_get_params[hid_map])
        assert not np.allclose(weights_straight_2.detach().numpy(), weights_get_params[out_map])


@pytest.mark.pytorch
@pytest.mark.accorrectness
class TestTrainingCorrectness:

    # test whether xor model created as autodiff composition learns properly
    @pytest.mark.parametrize(
        'eps, calls, opt, from_pnl_or_no', [
            (20000, 'single', 'adam', True),
            # (6000, 'multiple', 'adam', True),
            (20000, 'single', 'adam', False),
            # (6000, 'multiple', 'adam', False)
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                    pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                    ])
    def test_xor_training_correctness(self, eps, calls, opt, from_pnl_or_no,mode):
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

        xor = AutodiffComposition(param_init_from_pnl=from_pnl_or_no,
                                  optimizer_type=opt,
                                  learning_rate=0.1)

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

        if calls == 'single':
            results = xor.run(inputs={"inputs": {xor_in:xor_inputs},
                                      "targets": {xor_out:xor_targets},
                                      "epochs": eps}, bin_execute = mode
                                      )

            for i in range(len(results[0])):
                assert np.allclose(np.round(results[0][i][0]), xor_targets[i])

        else:
            results = xor.run(inputs={"inputs": {xor_in:xor_inputs},
                                      "targets": {xor_out:xor_targets},
                                      "epochs": 1}, bin_execute = mode
                                      )

            for i in range(eps-1):
                results = xor.run(inputs={"inputs": {xor_in:xor_inputs},
                                      "targets": {xor_out:xor_targets},
                                      "epochs": 1}, bin_execute = mode
                                      )

            for i in range(len(results[eps-1])):
                assert np.allclose(np.round(results[eps-1][i][0]), xor_targets[i])


    @pytest.mark.benchmark(group="Recurrent")
    # tests whether semantic network created as autodiff composition learns properly
    @pytest.mark.parametrize(
        'eps, opt, from_pnl_or_no', [
            (1000, 'adam', True),
            # (1000, 'adam', False)
        ]
    )
    @pytest.mark.parametrize("mode", ["Python",
                                    pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                    ])
    def test_semantic_net_training_correctness(self, eps, opt, from_pnl_or_no,mode,benchmark):

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
        sem_net = AutodiffComposition(param_init_from_pnl=from_pnl_or_no,
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
        result = sem_net.run(inputs={'inputs': inputs_dict,
                                      'targets': targets_dict,
                                      'epochs': eps}, bin_execute=mode)

        # CHECK CORRECTNESS
        for i in range(len(result[0])): # go over trial outputs in the single results entry
            for j in range(len(result[0][i])): # go over outputs for each output layer

                # get target for terminal node whose output state corresponds to current output
                correct_value = None
                curr_CIM_input_port = sem_net.output_CIM.input_ports[j]
                for output_state in sem_net.output_CIM_states.keys():
                    if sem_net.output_CIM_states[output_state][0] == curr_CIM_input_port:
                        node = output_state.owner
                        correct_value = targets_dict[node][i]

                # compare model output for terminal node on current trial with target for terminal node on current trial
                assert np.allclose(np.round(result[0][i][j]), correct_value)
        
        benchmark(sem_net.run,inputs={'inputs': inputs_dict,
                                      'targets': targets_dict,
                                      'epochs': eps}, bin_execute=mode)

    def test_pytorch_equivalence(self):
        from torch import nn, Tensor
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
                                   patience=patience,
                                   min_delta=min_delt,
                                   learning_rate=learning_rate,
                                   learning_enabled=True)

        mnet.add_node(il)
        mnet.add_node(cl)
        mnet.add_node(hl)
        mnet.add_node(ol)
        mnet.add_projection(projection=pih, sender=il, receiver=hl)
        mnet.add_projection(projection=pch, sender=cl, receiver=hl)
        mnet.add_projection(projection=pco, sender=cl, receiver=ol)
        mnet.add_projection(projection=pho, sender=hl, receiver=ol)

        mnet.run(
            inputs=input_set,
            minibatch_size=1,
        )
        mnet.learning_enabled = False

        mnet.run(
            inputs=input_set
        )

        loss = nn.MSELoss(reduction='sum')
        labels = Tensor(oSs.reshape(225))
        output = Tensor(mnet.parameters.results.get(mnet)[-15:]).reshape(225)

        assert np.allclose(loss(labels, output), 26.765404)

        comparator = Tensor([
            6.1453427e-03, 1.0000000e+00, 4.3089520e-03, 4.6516648e-03,
            8.2618001e-08, 7.6950570e-03, 1.0012715e-02, 7.4602116e-04,
            5.5932072e-03, 8.2061728e-03, 1.3183967e-02, 6.8695140e-03,
            2.0386763e-03, 1.6051590e-02, 4.4970796e-03, 2.5264964e-02,
            9.9999952e-01, 1.2149608e-02, 1.2654287e-02, 8.4562558e-08,
            2.1860752e-02, 3.0164845e-02, 1.7228123e-03, 1.6270705e-02,
            2.3551809e-02, 4.0039439e-02, 1.9407738e-02, 5.5244947e-03,
            5.3207204e-02, 1.2475645e-02, 1.8263543e-02, 9.9999857e-01,
            1.3606172e-02, 1.4083516e-02, 1.9825113e-07, 2.0812556e-02,
            9.8510243e-02, 3.6772678e-04, 2.3218473e-02, 2.1310158e-02,
            4.1438844e-02, 1.9753182e-02, 1.4779154e-04, 5.3911228e-02,
            1.4016392e-02, 2.2770427e-02, 9.9999523e-01, 1.6785931e-02,
            1.7425330e-02, 8.5163043e-07, 2.7639847e-02, 3.6300030e-02,
            3.1191204e-03, 2.1499839e-02, 2.9859981e-02, 4.6279125e-02,
            2.4875764e-02, 8.4926197e-03, 5.8906939e-02, 1.7091349e-02,
            2.5059190e-02, 9.9999011e-01, 1.8544273e-02, 1.9354496e-02,
            1.0527428e-06, 3.0156212e-02, 3.8953356e-02, 3.7354573e-03,
            2.3501266e-02, 3.2080259e-02, 4.9301058e-02, 4.3696441e-02,
            9.6132793e-03, 6.1628751e-02, 1.8898886e-02, 2.1750819e-02,
            9.9999368e-01, 1.6748825e-02, 1.7393829e-02, 7.1830226e-07,
            2.5006970e-02, 2.7698321e-02, 3.3105435e-03, 2.0801542e-02,
            2.5950141e-02, 3.3558644e-02, 2.2958852e-02, 2.0256745e-02,
            2.3319282e-01, 1.7054005e-02, 1.8514888e-02, 9.9999905e-01,
            1.3226544e-02, 1.3782033e-02, 2.6026042e-11, 4.6064097e-02,
            3.1815756e-02, 1.9899406e-03, 1.7096445e-02, 2.5150523e-02,
            4.1936234e-02, 1.9550020e-02, 5.8615361e-03, 5.4353956e-02,
            1.3454205e-02, 2.6853021e-02, 9.9997890e-01, 2.0612849e-02,
            2.1297636e-02, 3.8647496e-10, 6.3426509e-02, 4.1563153e-02,
            4.5182779e-03, 2.5162309e-02, 3.4463432e-02, 5.1794391e-02,
            2.7855596e-02, 1.0657175e-02, 6.3647740e-02, 2.0887125e-02,
            2.0779613e-02, 9.9999750e-01, 1.5266245e-02, 1.5672959e-02,
            3.2588846e-07, 2.3561811e-02, 3.4370810e-02, 2.5927520e-03,
            1.9366659e-02, 6.3743673e-02, 4.4676617e-02, 2.2506185e-02,
            7.1200798e-03, 5.6437019e-02, 1.5500024e-02, 1.9989355e-02,
            9.9999821e-01, 1.4523782e-02, 1.5081550e-02, 2.4768514e-07,
            2.3564184e-02, 3.3220161e-02, 2.3716728e-03, 1.8615939e-02,
            2.3655901e-02, 4.3229431e-02, 2.1647181e-02, 3.9079070e-02,
            5.5960458e-02, 1.4844272e-02, 2.6667269e-02, 9.9996638e-01,
            2.1370204e-02, 2.1930600e-02, 3.0933686e-06, 2.9115165e-02,
            1.2657930e-01, 8.4896327e-04, 3.4305997e-02, 2.9406046e-02,
            5.1382497e-02, 2.8318554e-02, 2.7292687e-04, 6.3303888e-02,
            2.1923987e-02, 1.8142583e-02, 9.9999869e-01, 1.3500703e-02,
            1.3975278e-02, 1.8918573e-07, 2.0690627e-02, 9.8083377e-02,
            3.6229755e-04, 2.3060752e-02, 2.1189153e-02, 4.1289508e-02,
            1.9628568e-02, 1.4622087e-04, 5.3754900e-02, 1.3907125e-02,
            1.6623152e-02, 9.9999928e-01, 1.2156339e-02, 1.2701786e-02,
            1.0281526e-07, 1.9703880e-02, 2.2795949e-02, 1.8327459e-03,
            1.5732812e-02, 2.0650970e-02, 2.8753385e-02, 1.7773779e-02,
            1.3212827e-02, 2.1228336e-01, 1.2412207e-02, 2.2249922e-02,
            9.9999511e-01, 1.6479362e-02, 1.7094113e-02, 5.8574432e-07,
            2.6067821e-02, 3.1332519e-02, 3.1530680e-03, 2.1082299e-02,
            2.7464284e-02, 1.6418625e-01, 2.3776721e-02, 8.3778230e-03,
            5.7937458e-02, 1.6737318e-02, 2.3775091e-02, 9.9998677e-01,
            1.8625231e-02, 1.9302633e-02, 1.3714227e-06, 2.7059222e-02,
            2.9545579e-02, 4.0283184e-03, 2.2817709e-02, 2.7989613e-02,
            3.5323191e-02, 2.4989001e-02, 2.3326760e-02, 2.4044928e-01,
            1.8946640e-02
        ])

        np.allclose(output,comparator,atol=1.e-6)

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
                                  num_trials=(eps*and_inputs.shape[0]+1))
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

        speedup = np.round((sys_time/comp_time), decimals=2)
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
                                  num_trials=(eps*xor_inputs.shape[0]+1))
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

        speedup = np.round((sys_time/comp_time), decimals=2)
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
                                  num_trials=(len(inputs_dict[nouns_in])*eps + 1))
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

        speedup = np.round((sys_time/comp_time), decimals=2)
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
            # (100, 'sgd')
        ]
    )
    def test_xor_training_identicalness(self, eps, opt):
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

        xor = AutodiffComposition(param_init_from_pnl=True,
                                  learning_rate=10,
                                  optimizer_type=opt
                                  )

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

        # TRAIN COMPOSITION
        inputs_dict = {"inputs": {xor_in:xor_inputs},
                       "targets": {xor_out:xor_targets},
                       "epochs": eps}

        result = xor.run(inputs=inputs_dict)
        comp_weights = xor.get_parameters()[0]

        # SET UP SYSTEM
        xor_process = Process(pathway=[xor_in_sys,
                                       hid_map_sys,
                                       xor_hid_sys,
                                       out_map_sys,
                                       xor_out_sys],
                              learning=pnl.LEARNING)
        xor_sys = System(processes=[xor_process],
                         learning_rate=10)

        # TRAIN SYSTEM
        results_sys = xor_sys.run(inputs={xor_in_sys:xor_inputs},
                                  targets={xor_out_sys:xor_targets},
                                  num_trials=(eps*xor_inputs.shape[0]))
        # CHECK THAT PARAMETERS FOR COMPOSITION, SYSTEM ARE SAME

        assert np.allclose(comp_weights[hid_map], hid_map_sys.get_mod_matrix(xor_sys))
        assert np.allclose(comp_weights[out_map], out_map_sys.get_mod_matrix(xor_sys))

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

        targets_dict_sys = {}
        targets_dict_sys[out_sig_I_sys] = targets_dict[out_sig_I]
        targets_dict_sys[out_sig_is_sys] = targets_dict[out_sig_is]
        targets_dict_sys[out_sig_has_sys] = targets_dict[out_sig_has]
        targets_dict_sys[out_sig_can_sys] = targets_dict[out_sig_can]

        sem_net.learning_enabled=False
        result = sem_net.run(inputs=inputs_dict)

        # comp_weights = sem_net.get_parameters()[0]

        # TRAIN COMPOSITION
        sem_net.learning_enabled=True

        result = sem_net.run(inputs={"inputs": inputs_dict,
                                     "targets": targets_dict,
                                     "epochs": eps}
                             )

        comp_weights = sem_net.get_parameters()[0]

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
                             learning_rate=0.5)

        # TRAIN SYSTEM

        results = sem_net_sys.run(inputs=inputs_dict_sys,
                                  targets=targets_dict_sys,
                                  num_trials=(len(inputs_dict_sys[nouns_in_sys])*eps))

        # CHECK THAT PARAMETERS FOR COMPOSITION, SYSTEM ARE SAME

        assert np.allclose(comp_weights[map_nouns_h1], map_nouns_h1_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_rels_h2], map_rels_h2_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h1_h2], map_h1_h2_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_I], map_h2_I_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_is], map_h2_is_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_has], map_h2_has_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(comp_weights[map_h2_can], map_h2_can_sys.get_mod_matrix(sem_net_sys))

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
        num_epochs = 10
        xor.run(inputs={"inputs": {xor_in: xor_inputs},
                        "targets": {xor_out: xor_targets},
                        "epochs": num_epochs}, do_logging=True)

        exec_id = xor.default_execution_id

        in_np_dict_vals = xor_in.log.nparray_dictionary()[exec_id]['value']
        in_np_vals = xor_in.log.nparray()[1][1][4][1:]

        hid_map_np_dict_mats = hid_map.log.nparray_dictionary()[exec_id]['matrix']
        hid_map_np_mats = np.array(hid_map.log.nparray()[1][1][4][1:])

        hid_np_dict_vals = xor_hid.log.nparray_dictionary()[exec_id]['value']

        out_map_np_dict_mats = out_map.log.nparray_dictionary()[exec_id]['matrix']
        out_map_np_mats = np.array(out_map.log.nparray()[1][1][4][1:])

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
            (2000, 4, 10, .00001),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                    pytest.param('LLVMRun', marks=[pytest.mark.llvm,pytest.mark.skip]), # Not implemented
                                    ])
    def test_xor_nested_train_then_no_train(self, num_epochs, learning_rate, patience, min_delta,mode):
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
            patience=patience,
            min_delta=min_delta,
            learning_rate=learning_rate,
            randomize=False,
            learning_enabled=True
        )

        xor_autodiff.add_node(xor_in)
        xor_autodiff.add_node(xor_hid)
        xor_autodiff.add_node(xor_out)

        xor_autodiff.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor_autodiff.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        # -----------------------------------------------------------------

        no_training_input_dict = {xor_in: xor_inputs}
        input_dict = {'inputs': {xor_in: xor_inputs}, 'targets': {xor_out: xor_targets}, 'epochs': num_epochs}

        parentComposition = pnl.Composition()
        parentComposition.add_node(xor_autodiff)
        input = {xor_autodiff: input_dict}
        no_training_input = {xor_autodiff: no_training_input_dict}

        result1 = parentComposition.run(inputs=input)

        xor_autodiff.learning_enabled = False
        result2 = parentComposition.run(inputs=no_training_input,bin_execute=mode)

        assert np.allclose(result1, [[0]], atol=0.1)
        assert np.allclose(result2, [[0]], atol=0.1)

    @pytest.mark.parametrize(
        'num_epochs, learning_rate, patience, min_delta', [
            (2000, 4, 10, .00001),
        ]
    )
    def test_xor_nested_no_train_then_train(self, num_epochs, learning_rate, patience, min_delta):
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
            patience=patience,
            min_delta=min_delta,
            learning_rate=learning_rate,
            randomize=False,
            learning_enabled=False
        )

        xor_autodiff.add_node(xor_in)
        xor_autodiff.add_node(xor_hid)
        xor_autodiff.add_node(xor_out)

        xor_autodiff.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor_autodiff.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        # -----------------------------------------------------------------

        no_training_input_dict = {xor_in: xor_inputs}
        input_dict = {'inputs': {xor_in: xor_inputs}, 'targets': {xor_out: xor_targets}, 'epochs': num_epochs}

        parentComposition = pnl.Composition()
        parentComposition.add_node(xor_autodiff)
        input = {xor_autodiff: input_dict}
        no_training_input = {xor_autodiff: no_training_input_dict}

        result1 = parentComposition.run(inputs=no_training_input)

        xor_autodiff.learning_enabled = True
        result2 = parentComposition.run(inputs=input)

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
    def test_semantic_net_nested(self, eps, opt):

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

        parentComposition.run(inputs=input)

        sem_net.learning_enabled = False

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

        xor.run(
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

        xor.run(
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

        xor.run(
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
            xor.run(
                inputs=inputs_dict_2,
                context=c2,
                minibatch_size=TRAINING_SET
            )
            xor.run(
                inputs=inputs_dict_3,
                context=c2,
                minibatch_size=TRAINING_SET
            )

        c1_results = xor.parameters.results._get(c1)
        c2_results = xor.parameters.results._get(c2)

        assert np.allclose(c1_results[0][:2], c2_results[-2])
        assert np.allclose(c1_results[0][2:], c2_results[-1])

