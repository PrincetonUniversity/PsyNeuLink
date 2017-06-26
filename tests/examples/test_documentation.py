import numpy as np

from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Functions.Function import THRESHOLD
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import RecurrentTransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import LEARNING_PROJECTION, NOISE, PARAMETER_STATE_PARAMS, RANDOM_CONNECTIVITY_MATRIX
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import WhenFinished


class TestDocumentationExamples:

    class TestDynamics:

        def test_settling_threshold(self):
            my_input_layer = TransferMechanism(size=3)
            my_recurrent_layer = RecurrentTransferMechanism(size=10)
            my_response_layer = TransferMechanism(size=3)
            settling_process = process(pathway=[my_input_layer, my_recurrent_layer, my_response_layer])

            settling_system = system(
                processes=[settling_process]
            )

            settling_system.scheduler_processing = Scheduler(system=settling_system)
            settling_system.scheduler_processing.add_condition(my_response_layer, WhenFinished(my_recurrent_layer))

            settling_system.run()

    class TestMisc:

        def test_mechs_in_pathway(self):
            mechanism_1 = TransferMechanism()
            mechanism_2 = DDM()
            some_params = {PARAMETER_STATE_PARAMS: {THRESHOLD: 2, NOISE: 0.1}}
            my_process = process(pathway=[mechanism_1, TransferMechanism, (mechanism_2, some_params)])
            result = my_process.execute()
            # changed result from 2 to 1
            assert(result == np.array([1]))

        def test_default_projection(self):
            mechanism_1 = TransferMechanism()
            mechanism_2 = TransferMechanism()
            mechanism_3 = DDM()
            my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
            result = my_process.execute()

            assert(result == np.array([1.]))

        def test_inline_projection_using_existing_projection(self):
            mechanism_1 = TransferMechanism()
            mechanism_2 = TransferMechanism()
            mechanism_3 = DDM()
            projection_A = MappingProjection()
            my_process = process(pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])
            result = my_process.execute()

            assert(result == np.array([1.]))

        def test_inline_projection_using_keyword(self):
            mechanism_1 = TransferMechanism()
            mechanism_2 = TransferMechanism()
            mechanism_3 = DDM()
            my_process = process(pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2, mechanism_3])
            result = my_process.execute()

            assert(result == np.array([1.]))

        def test_standalone_projection(self):
            mechanism_1 = TransferMechanism()
            mechanism_2 = TransferMechanism()
            mechanism_3 = DDM()
            projection_A = MappingProjection(sender=mechanism_1, receiver=mechanism_2)
            my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
            result = my_process.execute()

            assert(result == np.array([1.]))

        def test_process_learning(self):
            mechanism_1 = TransferMechanism(function=Logistic)
            mechanism_2 = TransferMechanism(function=Logistic)
            mechanism_3 = TransferMechanism(function=Logistic)
            my_process = process(
                pathway=[mechanism_1, mechanism_2, mechanism_3],
                learning=LEARNING_PROJECTION,
                target=[0],
            )
            result = my_process.execute()

            np.testing.assert_allclose(result, np.array([0.65077768]))
