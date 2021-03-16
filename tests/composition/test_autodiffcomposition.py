import logging
import timeit as timeit

import numpy as np

import pytest

import psyneulink as pnl

from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.components.functions.learningfunctions import BackPropagation
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals import Context
from psyneulink.core.globals.keywords import TRAINING_SET
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition

logger = logging.getLogger(__name__)


# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for functions of AutodiffComposition class that are new (not in Composition)
# or override functions in Composition

@pytest.mark.pytorch
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                 ])
def test_autodiff_forward(mode):
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
    xor = AutodiffComposition()

    xor.add_node(xor_in)
    xor.add_node(xor_hid)
    xor.add_node(xor_out)

    xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
    xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

    outputs = xor.run(inputs=[0,0], bin_execute=mode)
    assert np.allclose(outputs, [[0.9479085241082691]])

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
    def test_weight_initialization(self):

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
        xor = AutodiffComposition()

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

        xor = AutodiffComposition()

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

        xor = AutodiffComposition(loss_spec=loss)

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

        xor = AutodiffComposition(loss_spec=ls)

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
        'learning_rate, weight_decay, optimizer_type, expected', [
            (10, 0, 'sgd', [[[0.9863038667851067]], [[0.9944287263151904]], [[0.9934801466163382]], [[0.9979153035411085]]]),
            (1.5, 1, 'sgd', None),
            (1.5, 1, 'adam', None),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    def test_optimizer_specs(self, learning_rate, weight_decay, optimizer_type, expected, mode, benchmark):
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

        xor = AutodiffComposition(learning_rate=learning_rate,
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

        if expected is not None:
            assert np.allclose(results_before_proc, expected)

        if benchmark.enabled:
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

        xor = AutodiffComposition(learning_rate=10.0,
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

@pytest.mark.pytorch
@pytest.mark.accorrectness
class TestTrainingCorrectness:

    # test whether xor model created as autodiff composition learns properly
    @pytest.mark.benchmark(group="XOR")
    @pytest.mark.parametrize(
        'eps, calls, opt, expected', [
            (100, 'single', 'adam', [[[0.09823965]], [[0.81092879]], [[0.78179557]], [[0.25593583]]]),
            (50, 'multiple', 'adam', [[[0.31200036]], [[0.59406178]], [[0.60417587]], [[0.52347365]]]),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    def test_xor_training_correctness(self, eps, calls, opt, mode, benchmark, expected):
        xor_in = TransferMechanism(name='xor_in',
                                   default_variable=np.zeros(2))

        xor_hid = TransferMechanism(name='xor_hid',
                                    default_variable=np.zeros(10),
                                    function=Logistic())

        xor_out = TransferMechanism(name='xor_out',
                                    default_variable=np.zeros(1),
                                    function=Logistic())

        hid_map = MappingProjection(matrix=np.random.rand(2, 10))
        out_map = MappingProjection(matrix=np.random.rand(10, 1))

        xor = AutodiffComposition(optimizer_type=opt,
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

        else:
            input_dict = {"inputs": {xor_in: xor_inputs},
                          "targets": {xor_out: xor_targets},
                          "epochs": 1}
            for i in range(eps):
                results = xor.learn(inputs=input_dict, bin_execute=mode)

        assert len(results) == len(expected)
        for r, t in zip(results, expected):
            assert np.allclose(r[0], t)

        if benchmark.enabled:
            benchmark(xor.learn, inputs={"inputs": {xor_in: xor_inputs},
                                         "targets": {xor_out: xor_targets},
                                         "epochs": eps}, bin_execute=mode)


    # tests whether semantic network created as autodiff composition learns properly
    @pytest.mark.benchmark(group="Semantic net")
    @pytest.mark.parametrize(
        'eps, opt', [
            (50, 'adam'),
        ]
    )
    @pytest.mark.parametrize("mode", ['Python',
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                     ])
    def test_semantic_net_training_correctness(self, eps, opt, mode, benchmark):

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
        sem_net = AutodiffComposition(optimizer_type=opt, learning_rate=.001)

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
        results = sem_net.learn(inputs={'inputs': inputs_dict,
                                        'targets': targets_dict,
                                        'epochs': eps}, bin_execute=mode)

        # CHECK CORRECTNESS
        expected = [[[0.13455769, 0.12924714, 0.13288172, 0.1404659 , 0.14305814,
                    0.14801862, 0.13968417, 0.14191982], [0.99991336, 0.99989676, 0.50621126, 0.52845353, 0.25829169,
                    0.26210662, 0.24857125, 0.25041139, 0.4960792 , 0.25689092,
                    0.36730921, 0.24619036], [0.5026992 , 0.51272669, 0.24175637, 0.25034784, 0.2406316 ,
                    0.23613707, 0.25192136, 0.25169522, 0.25615122], [0.99931055, 0.50694136, 0.23105683, 0.24641822, 0.49398116,
                    0.25462159, 0.24433084, 0.24267716, 0.52209116]], [[0.13434049, 0.14280817, 0.13996269, 0.15002735, 0.12975428,
                    0.13143489, 0.14087601, 0.1384075 ], [0.99988782, 0.99987699, 0.50700037, 0.48324852, 0.23605196,
                    0.24623233, 0.25029056, 0.23937042, 0.51119504, 0.2428462 ,
                    0.38335829, 0.24875281], [0.51938115, 0.48541386, 0.25513764, 0.26880953, 0.27165005,
                    0.2768918 , 0.25270764, 0.25394743, 0.2453564 ], [0.99932127, 0.48854068, 0.25824899, 0.24792285, 0.52373987,
                    0.23226666, 0.26078735, 0.25531502, 0.48945013]], [[0.14603474, 0.13076473, 0.14285659, 0.14135699, 0.13505543,
                    0.13515591, 0.14114513, 0.13170369], [0.99990678, 0.99989262, 0.48366278, 0.49653231, 0.24862611,
                    0.23489764, 0.25762623, 0.2557183 , 0.48931577, 0.24042071,
                    0.38135237, 0.2559529 ], [0.48125071, 0.49795935, 0.25190272, 0.23454573, 0.24576829,
                    0.24084843, 0.23960781, 0.24709719, 0.24797553], [0.99927591, 0.50571286, 0.25989537, 0.25337519, 0.49977912,
                    0.2629161 , 0.24592842, 0.25430635, 0.48884817]], [[0.13471359, 0.12938359, 0.12833093, 0.13589542, 0.14362713,
                    0.15158585, 0.14307399, 0.14359844], [0.99991663, 0.99990086, 0.50239329, 0.5363237 , 0.26273315,
                    0.26079003, 0.24686505, 0.2514966 , 0.50092905, 0.26094556,
                    0.36939329, 0.24278735], [0.50633107, 0.51005404, 0.24486936, 0.25305515, 0.2371648 ,
                    0.23443371, 0.24958497, 0.25592672, 0.26077736], [0.99930883, 0.51096502, 0.2273585 , 0.24095976, 0.49420255,
                    0.2541994 , 0.24348927, 0.23870194, 0.52777643]], [[0.13437031, 0.14294567, 0.13462913, 0.14499767, 0.13014469,
                    0.1347783 , 0.14426174, 0.14003815], [0.99989275, 0.99988313, 0.50533481, 0.49070575, 0.24083416,
                    0.24419528, 0.24850006, 0.2388715 , 0.51590916, 0.2477704 ,
                    0.38492997, 0.24480191], [0.52312591, 0.48400766, 0.25827697, 0.27239575, 0.26774108,
                    0.27407821, 0.24982668, 0.25714894, 0.24895088], [0.9993246 , 0.49095133, 0.25329179, 0.2414722 , 0.52311016,
                    0.23199452, 0.25880536, 0.25034698, 0.49623819]], [[0.14624279, 0.13155149, 0.13771617, 0.13640541, 0.1355243 ,
                    0.13836647, 0.14442104, 0.13306906], [0.9999103 , 0.99989714, 0.48256079, 0.50278716, 0.25496791,
                    0.23361944, 0.25567964, 0.25577915, 0.49540433, 0.24632798,
                    0.38186044, 0.25220444], [0.48732786, 0.49713526, 0.25686322, 0.23903771, 0.24179882,
                    0.238869  , 0.2368224 , 0.25066201, 0.25139028], [0.9992754 , 0.50799072, 0.25448626, 0.24711122, 0.49775634,
                    0.26134282, 0.24471328, 0.24960873, 0.49669515]], [[0.13743409, 0.12655781, 0.13687131, 0.14444673, 0.14586922,
                    0.14982043, 0.14115407, 0.14425256], [0.999905  , 0.99988946, 0.5111958 , 0.52953632, 0.25829416,
                    0.25816645, 0.24772719, 0.2493831 , 0.49172307, 0.25647241,
                    0.36490702, 0.24579828], [0.49409604, 0.51876333, 0.2438246 , 0.25106641, 0.24575334,
                    0.23758284, 0.25036778, 0.24501611, 0.26348273], [0.99927212, 0.49900832, 0.23785606, 0.24568878, 0.49229404,
                    0.25799455, 0.2380617 , 0.24483316, 0.51242196]], [[0.13690143, 0.14031709, 0.14406563, 0.15349375, 0.13257716,
                    0.13310877, 0.14252974, 0.14056437], [0.99987794, 0.99986931, 0.5140066 , 0.48347409, 0.23776129,
                    0.24271912, 0.2491253 , 0.23773346, 0.50725307, 0.24412832,
                    0.37955962, 0.24840394], [0.5134345 , 0.49359454, 0.25820627, 0.27104423, 0.27620957,
                    0.27795555, 0.25014143, 0.24655665, 0.25137209], [0.99928701, 0.47923949, 0.26354622, 0.24637029, 0.51954436,
                    0.23441734, 0.25357643, 0.25643598, 0.48217473]], [[0.14885265, 0.12888511, 0.14747413, 0.14492694, 0.13757074,
                    0.13653999, 0.14230078, 0.1333605 ], [0.99989882, 0.99988608, 0.49175208, 0.49464422, 0.25033443,
                    0.2315963 , 0.25565926, 0.25286519, 0.48521686, 0.24180012,
                    0.37838253, 0.25482498], [0.47632935, 0.50754291, 0.25540722, 0.23699234, 0.24989352,
                    0.24130117, 0.236814  , 0.23865942, 0.25337658], [0.99924137, 0.49439904, 0.26500386, 0.25133608, 0.49471845,
                    0.26444086, 0.23836089, 0.2550785 , 0.48265071]], [[0.13796428, 0.13122133, 0.13665485, 0.14238789, 0.14646363,
                    0.14953325, 0.14001922, 0.14631467], [0.99990658, 0.99988831, 0.51044886, 0.52877409, 0.26162535,
                    0.26113322, 0.24464028, 0.25460667, 0.50142119, 0.26075076,
                    0.36249296, 0.2469639 ], [0.49888045, 0.520344  , 0.24213434, 0.25386651, 0.23856703,
                    0.23283943, 0.25382764, 0.25348876, 0.25930602], [0.9992759 , 0.50552098, 0.23581279, 0.25056642, 0.49046455,
                    0.26042018, 0.24011749, 0.24306269, 0.52240185]], [[0.13750907, 0.14541291, 0.1443253 , 0.15189612, 0.13295338,
                    0.13279175, 0.14102944, 0.14247126], [0.99987945, 0.99986745, 0.51385772, 0.48177978, 0.23992435,
                    0.24651741, 0.24564878, 0.2423684 , 0.51605876, 0.24748102,
                    0.37804356, 0.24958671], [0.51825942, 0.49563748, 0.25600593, 0.27323944, 0.26870607,
                    0.27273737, 0.25405544, 0.25468467, 0.24732677], [0.99928937, 0.4851184 , 0.26178837, 0.25136875, 0.51760249,
                    0.23648163, 0.25573403, 0.25496827, 0.49227648]], [[0.14930882, 0.13316748, 0.14767505, 0.14350067, 0.13804021,
                    0.13603459, 0.14084128, 0.13513649], [0.99990066, 0.99988511, 0.49193637, 0.49266043, 0.25237073,
                    0.23599757, 0.25202475, 0.25740929, 0.49266556, 0.24476454,
                    0.37618961, 0.25697362], [0.48210255, 0.51007883, 0.25286726, 0.2385758 , 0.24213314,
                    0.23616873, 0.23972905, 0.24636349, 0.24902741], [0.99924664, 0.4998954 , 0.2624679 , 0.25622887, 0.49190217,
                    0.26610712, 0.24000543, 0.25322914, 0.49359824]], [[0.13981779, 0.13517728, 0.13714904, 0.14312451, 0.14264037,
                    0.1471793 , 0.13872158, 0.14604209], [0.99990026, 0.99988117, 0.50978024, 0.51828715, 0.26366737,
                    0.26281186, 0.24835743, 0.25797436, 0.50770859, 0.26547003,
                    0.36335027, 0.24615446], [0.50366793, 0.51568697, 0.2470342 , 0.2533224 , 0.23632146,
                    0.23162517, 0.25117307, 0.25934756, 0.256024  ], [0.99924436, 0.50891146, 0.23908499, 0.24926875, 0.48979495,
                    0.26367751, 0.24330774, 0.24296292, 0.523776  ]], [[0.13961147, 0.14975401, 0.14472653, 0.15355384, 0.12925548,
                    0.13046081, 0.13938395, 0.14208934], [0.99987039, 0.99985783, 0.51256916, 0.47090975, 0.24108525,
                    0.24849742, 0.24950535, 0.24553023, 0.522052  , 0.25148739,
                    0.37924791, 0.24931395], [0.52265254, 0.49040645, 0.26115407, 0.27231193, 0.26698365,
                    0.27201252, 0.252447  , 0.26033848, 0.24417141], [0.99925421, 0.48866289, 0.26591473, 0.25072477, 0.51818878,
                    0.23925897, 0.26008435, 0.25565447, 0.49313837]], [[0.15131984, 0.13702244, 0.14796541, 0.14499107, 0.13516231,
                    0.13379344, 0.13967548, 0.13504357], [0.99989266, 0.99987644, 0.49069576, 0.48284025, 0.25373136,
                    0.23780959, 0.25705167, 0.26031535, 0.49734223, 0.24868116,
                    0.37609291, 0.25826094], [0.48615805, 0.50439585, 0.25764469, 0.23792093, 0.24102865,
                    0.23622676, 0.23835519, 0.25200694, 0.24543424], [0.99920779, 0.50378559, 0.265772  , 0.25635798, 0.49252096,
                    0.26892098, 0.24458074, 0.25443214, 0.494262  ]], [[0.13850469, 0.13222947, 0.13670674, 0.14224276, 0.14481382,
                    0.14983366, 0.14173916, 0.14323358], [0.99990126, 0.99988249, 0.51196132, 0.52177217, 0.2678038 ,
                    0.26687693, 0.24660127, 0.25415374, 0.50244534, 0.26499807,
                    0.36556967, 0.24445311], [0.51065084, 0.52060372, 0.25297718, 0.25785385, 0.24041214,
                    0.23595817, 0.24912937, 0.25285752, 0.26092579], [0.9992282 , 0.49866166, 0.23316642, 0.2414636 , 0.48397547,
                    0.25549083, 0.24296723, 0.24121413, 0.52743639]], [[0.13795671, 0.14608656, 0.14389815, 0.15250863, 0.13137121,
                    0.13292996, 0.14235728, 0.13916877], [0.99987232, 0.99986024, 0.514095  , 0.47542769, 0.24452563,
                    0.25126078, 0.248764  , 0.241217  , 0.51579474, 0.25047855,
                    0.38175049, 0.24768651], [0.52795513, 0.49433932, 0.26639626, 0.2765692 , 0.27249335,
                    0.27735353, 0.25105699, 0.25316897, 0.24833112], [0.99924257, 0.47881092, 0.25918246, 0.24338929, 0.51375801,
                    0.23187301, 0.26019419, 0.25420934, 0.49555353]], [[0.14984495, 0.13387861, 0.14722304, 0.14355772, 0.13724767,
                    0.13665399, 0.14239016, 0.1320989 ], [0.99989382, 0.9998781 , 0.49129168, 0.48786117, 0.25736056,
                    0.24076914, 0.2571532 , 0.25661644, 0.49134559, 0.24762313,
                    0.37894905, 0.25678541], [0.49115594, 0.50720606, 0.26343239, 0.24118617, 0.24666918,
                    0.24228136, 0.23828715, 0.24556003, 0.24986447], [0.9991909 , 0.49609669, 0.25952719, 0.2503092 , 0.48831756,
                    0.26050785, 0.24624062, 0.25460693, 0.49571571]], [[0.13798156, 0.13536763, 0.13556549, 0.14026479, 0.14613495,
                    0.1517147 , 0.14150378, 0.14582109], [0.99990224, 0.99988499, 0.51321887, 0.5234933 , 0.26596025,
                    0.26627399, 0.24845062, 0.25224455, 0.50060705, 0.26840742,
                    0.36382243, 0.24724158], [0.50857345, 0.52097167, 0.24807872, 0.25421915, 0.23896201,
                    0.23521938, 0.25167447, 0.25399296, 0.25680194], [0.99924088, 0.50538628, 0.23168769, 0.24466784, 0.48374493,
                    0.25836643, 0.24395082, 0.24271583, 0.52908765]], [[0.13738911, 0.14964169, 0.14240066, 0.14996998, 0.13258046,
                    0.13529629, 0.14257416, 0.14196438], [0.99987348, 0.99986355, 0.51547364, 0.47846669, 0.24280831,
                    0.25045232, 0.25106236, 0.23941227, 0.51388317, 0.25405565,
                    0.38069812, 0.25010755], [0.52466269, 0.49422476, 0.26072088, 0.27243928, 0.27097311,
                    0.27644019, 0.25390417, 0.25487688, 0.24468432], [0.99925402, 0.48653142, 0.25783738, 0.24674309, 0.51370588,
                    0.23544836, 0.26119428, 0.25606803, 0.49686813]], [[0.14925705, 0.13686349, 0.14561612, 0.14139895, 0.13823102,
                    0.13876809, 0.14286744, 0.13452957], [0.99989461, 0.99988054, 0.49152081, 0.49115061, 0.25525704,
                    0.23961845, 0.25940244, 0.25531285, 0.48988801, 0.25049358,
                    0.37898184, 0.25880442], [0.48716519, 0.50598831, 0.25786953, 0.23748522, 0.24522105,
                    0.24133955, 0.24092032, 0.247859  , 0.24660646], [0.99920254, 0.50397091, 0.25882576, 0.25383231, 0.48962604,
                    0.26477928, 0.24738806, 0.25637011, 0.49634854]], [[0.13466774, 0.13597299, 0.13044001, 0.13621353, 0.14063205,
                    0.14927389, 0.13741799, 0.14182897], [0.99991209, 0.99989375, 0.50664715, 0.52647149, 0.26631381,
                    0.26772046, 0.25065527, 0.25828284, 0.51113304, 0.26876027,
                    0.36519173, 0.24445213], [0.50981376, 0.51216926, 0.24469315, 0.25069097, 0.23575157,
                    0.23351051, 0.25738456, 0.26157098, 0.25154516], [0.9992887 , 0.51923276, 0.2286251 , 0.24647828, 0.48904116,
                    0.25948029, 0.25191475, 0.24172423, 0.53116312]], [[0.13456692, 0.15039528, 0.13737152, 0.14609287, 0.12736956,
                    0.13283867, 0.13853673, 0.13816526], [0.99988542, 0.99987272, 0.50746352, 0.48137383, 0.24278802,
                    0.25182679, 0.25260023, 0.24645783, 0.52561721, 0.25394353,
                    0.38280036, 0.24682469], [0.52522447, 0.48470641, 0.25775314, 0.26859876, 0.26712899,
                    0.27417417, 0.25948482, 0.26348026, 0.24082551], [0.9992959 , 0.5009347 , 0.25614941, 0.24856744, 0.51984942,
                    0.23715273, 0.2693644 , 0.2551465 , 0.49819368]], [[0.14562679, 0.13745423, 0.13996431, 0.13722137, 0.13278566,
                    0.13632886, 0.13899164, 0.1314741 ], [0.99990502, 0.99988943, 0.48410876, 0.49522954, 0.2553264 ,
                    0.24084323, 0.26058312, 0.26304712, 0.50254737, 0.2507632 ,
                    0.3802493 , 0.25486957], [0.48768232, 0.49615974, 0.25448472, 0.23400199, 0.24116754,
                    0.23886036, 0.24575353, 0.25715595, 0.24334699], [0.99925183, 0.51889063, 0.25712839, 0.25460613, 0.49597306,
                    0.26739429, 0.25464059, 0.25453138, 0.49761396]]]

        for res, exp in zip(results, expected):
            for r, e in zip(res, exp):
                assert np.allclose(r, e)
        if benchmark.enabled:
            benchmark(sem_net.learn, inputs={'inputs': inputs_dict,
                                             'targets': targets_dict,
                                             'epochs': eps}, bin_execute=mode)

    @pytest.mark.parametrize("mode", ['Python',
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

        mnet = AutodiffComposition(learning_rate=learning_rate)

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

        mnet = AutodiffComposition(learning_rate=learning_rate)

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
                                        matrix=and_map.matrix.base.copy(),
                                        sender=and_in_sys,
                                        receiver=and_out_sys)

        # SET UP COMPOSITION

        and_net = AutodiffComposition()

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

        msg = 'Training XOR model as AutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
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
                                        matrix=hid_map.matrix.base.copy(),
                                        sender=xor_in_sys,
                                        receiver=xor_hid_sys)

        out_map_sys = MappingProjection(name='out_map_sys',
                                        matrix=out_map.matrix.base.copy(),
                                        sender=xor_hid_sys,
                                        receiver=xor_out_sys)

        # SET UP COMPOSITION

        xor = AutodiffComposition(bin_execute=mode)

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

        # xor_process = Process(pathway=[xor_in_sys,
        #                                hid_map_sys,
        #                                xor_hid_sys,
        #                                out_map_sys,
        #                                xor_out_sys],
        #                       learning=pnl.LEARNING)

        xor_process = Composition(pathways=([xor_in_sys,
                                       hid_map_sys,
                                       xor_hid_sys,
                                       out_map_sys,
                                       xor_out_sys], BackPropagation))

        msg = 'Training XOR model as AutodiffComposition for {eps} epochs took {comp_time} seconds.'
        print(msg)
        print("\n")
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

        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.base.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)

        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.base.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)

        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.base.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)

        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.base.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)

        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.base.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)

        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.base.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)

        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.base.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)

        # COMPOSITION FOR SEMANTIC NET

        sem_net = AutodiffComposition()

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

        msg = 'Training Semantic net as AutodiffComposition for {0} epochs took {1} seconds'.format(eps, comp_time)
        print(msg)
        print("\n")
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

        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.base.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)

        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.base.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)

        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.base.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)

        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.base.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)

        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.base.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)

        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.base.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)

        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.base.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)

        # SET UP COMPOSITION FOR SEMANTIC NET
        sem_net = AutodiffComposition(learning_rate=0.5,
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

        # TRAIN COMPOSITION
        def g_f():
            yield {"inputs": inputs_dict,
                   "targets": targets_dict,
                   "epochs": eps}
        g = g_f()
        result = sem_net.learn(inputs=g_f)

        # SET UP SYSTEM
        sem_net_sys = Composition()

        backprop_pathway = sem_net_sys.add_backpropagation_learning_pathway(
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
        inputs_dict_sys[backprop_pathway.target] = targets_dict[out_sig_I]

        backprop_pathway = sem_net_sys.add_backpropagation_learning_pathway(
            pathway=[
                rels_in_sys,
                map_rels_h2_sys,
                h2_sys,
                map_h2_is_sys,
                out_sig_is_sys
            ],
            learning_rate=0.5
        )
        inputs_dict_sys[backprop_pathway.target] = targets_dict[out_sig_is]

        backprop_pathway = sem_net_sys.add_backpropagation_learning_pathway(
            pathway=[
                h2_sys,
                map_h2_has_sys,
                out_sig_has_sys
            ],
            learning_rate=0.5
        )
        inputs_dict_sys[backprop_pathway.target] = targets_dict[out_sig_has]

        backprop_pathway = sem_net_sys.add_backpropagation_learning_pathway(
            pathway=[
                h2_sys,
                map_h2_can_sys,
                out_sig_can_sys
            ],
            learning_rate=0.5
        )
        inputs_dict_sys[backprop_pathway.target] = targets_dict[out_sig_can]

        # TRAIN SYSTEM
        results = sem_net_sys.learn(inputs=inputs_dict_sys,
                                  num_trials=(len(inputs_dict_sys[nouns_in_sys]) * eps))

        # CHECK THAT PARAMETERS FOR COMPOSITION, SYSTEM ARE SAME

        assert np.allclose(map_nouns_h1.parameters.matrix.get(sem_net), map_nouns_h1_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(map_rels_h2.parameters.matrix.get(sem_net), map_rels_h2_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(map_h1_h2.parameters.matrix.get(sem_net), map_h1_h2_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(map_h2_I.parameters.matrix.get(sem_net), map_h2_I_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(map_h2_is.parameters.matrix.get(sem_net), map_h2_is_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(map_h2_has.parameters.matrix.get(sem_net), map_h2_has_sys.get_mod_matrix(sem_net_sys))
        assert np.allclose(map_h2_can.parameters.matrix.get(sem_net), map_h2_can_sys.get_mod_matrix(sem_net_sys))

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

        xor_dict = AutodiffComposition()

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

        xor_func = AutodiffComposition()

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

        def get_inputs(idx):
            return {
                "inputs": {
                    xor_in_func: xor_inputs_func[idx]
                },
                "targets": {
                    xor_out_func: xor_targets_func[idx]
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

        xor_gen = AutodiffComposition()

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

        xor_gen_func = AutodiffComposition()

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

        xor = AutodiffComposition()

        xor.add_node(xor_in)
        xor.add_node(xor_hid)
        xor.add_node(xor_out)

        xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

        xor_in.set_log_conditions('value', pnl.LogCondition.TRIAL)
        xor_hid.set_log_conditions('value', pnl.LogCondition.TRIAL)
        xor_out.set_log_conditions('value', pnl.LogCondition.TRIAL)

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

    def test_autodiff_loss_tracking(self):
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

        xor = AutodiffComposition()

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
        num_epochs = 100
        xor.learn(inputs={"inputs": {xor_in: xor_inputs},
                        "targets": {xor_out: xor_targets},
                        "epochs": num_epochs})

        losses = xor.losses
        # Since the losses track average losses per weight update, and weights are updated every minibatch,
        # and minibatch_size is 1, then there should be num_epochs * num_minibatches = num_epochs * 4
        # total entries
        expected_loss_length = num_epochs * len(xor_inputs)
        assert len(losses) == expected_loss_length

        # test clearing ad losses
        xor.clear_losses(context=xor)
        assert len(xor.losses) == 0

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
            learning_rate=learning_rate,
        )

        xor_autodiff.add_node(xor_in)
        xor_autodiff.add_node(xor_hid)
        xor_autodiff.add_node(xor_out)

        xor_autodiff.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
        xor_autodiff.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

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
            learning_rate=learning_rate,
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

        map_nouns_h1_sys = MappingProjection(matrix=map_nouns_h1.matrix.base.copy(),
                                             name="map_nouns_h1_sys",
                                             sender=nouns_in_sys,
                                             receiver=h1_sys)

        map_rels_h2_sys = MappingProjection(matrix=map_rels_h2.matrix.base.copy(),
                                        name="map_relh2_sys",
                                        sender=rels_in_sys,
                                        receiver=h2_sys)

        map_h1_h2_sys = MappingProjection(matrix=map_h1_h2.matrix.base.copy(),
                                          name="map_h1_h2_sys",
                                          sender=h1_sys,
                                          receiver=h2_sys)

        map_h2_I_sys = MappingProjection(matrix=map_h2_I.matrix.base.copy(),
                                         name="map_h2_I_sys",
                                         sender=h2_sys,
                                         receiver=out_sig_I_sys)

        map_h2_is_sys = MappingProjection(matrix=map_h2_is.matrix.base.copy(),
                                          name="map_h2_is_sys",
                                          sender=h2_sys,
                                          receiver=out_sig_is_sys)

        map_h2_has_sys = MappingProjection(matrix=map_h2_has.matrix.base.copy(),
                                           name="map_h2_has_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_has_sys)

        map_h2_can_sys = MappingProjection(matrix=map_h2_can.matrix.base.copy(),
                                           name="map_h2_can_sys",
                                           sender=h2_sys,
                                           receiver=out_sig_can_sys)

        # SET UP COMPOSITION FOR SEMANTIC NET

        sem_net = AutodiffComposition(learning_rate=0.5,
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

        xor = AutodiffComposition(learning_rate=10)

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

        xor = AutodiffComposition(learning_rate=10)

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

        xor = AutodiffComposition(learning_rate=10,
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
