from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Globals.Keywords import *

run_DDM_tests = True
run_transfer_tests = False
run_distribution_test = False

if run_DDM_tests:
    print("DDM Test #1: Execute DDM with noise = 0.5")

    my_DDM = DDM(function=Integrator( integration_type = DIFFUSION, noise=0.5),
                name='My_DDM',
                time_scale=TimeScale.TIME_STEP
                )
    my_DDM.execute()

    print("-------------------------------------------------")

    print("DDM Test #2: Execute DDM with noise = NormalDist(mean=1.0, standard_dev = 0.5).function")
    print("(NOT Valid)")

    try:
        my_DDM2 = DDM(function=Integrator( integration_type=DIFFUSION, noise=NormalDist(mean=1.0, standard_dev=0.5).function),
                      name='My_DDM2',
                      time_scale=TimeScale.TIME_STEP
                      )
        my_DDM2.execute()
    except FunctionError:
        print("Passed")

    print("-------------------------------------------------")

    print("DDM Test #3: Execute DDM with noise not specified")
    my_DDM3 = DDM(function=Integrator( integration_type=DIFFUSION, ),
                  name='My_DDM3',
                  time_scale=TimeScale.TIME_STEP
                  )
    my_DDM3.execute()
    print("Passed")
    print("-------------------------------------------------")

    print("DDM Test #4: Execute DDM with noise = NormalDist(mean=1.0, standard_dev = 0.5)")
    print("(NOT Valid)")
    try:
        my_DDM4 = DDM(function=Integrator( integration_type=DIFFUSION, noise=NormalDist(mean=1.0, standard_dev=0.5)),
                      name='My_DDM4',
                      time_scale=TimeScale.TIME_STEP
                      )
        my_DDM4.execute()
    except FunctionError:
        print("Passed")
    print("-------------------------------------------------")

    print("DDM Test #5: Execute DDM with noise = 0.5; Repeat until a threshold of 30")

    threshold = 30
    position = 0
    while abs(position) < threshold:
        position = my_DDM.execute()[0][0]

    print("Passed")
    print("-------------------------------------------------")

    print("DDM Test #6: Execute DDM in trial mode")
    my_DDM6 = DDM(function=BogaczEtAl(drift_rate=3.0,
                                      threshold=30.0),
                  name='MY_DDM6'
                  )

    my_DDM6.execute([[10]])

    print("Passed")
    print("-------------------------------------------------")

    print("DDM Test #7: Execute DDM with noise = 0.5, integration_type=CONSTANT")
    print("(NOT Valid)")

    try:
        my_DDM7 = DDM(function=Integrator(noise=0.5, integration_type=CONSTANT),
                      name='My_DDM7',
                      time_scale=TimeScale.TIME_STEP
                      )
        my_DDM7.execute()
    except MechanismError:
        print("Passed")

    print("-------------------------------------------------")


if run_transfer_tests:
    print("Transfer Test #1: Execute Transfer with noise = 5")
    print("(NOT Valid)")


    try:
        my_Transfer_Test = TransferMechanism(name='my_Transfer_Test',
                               default_input_value = [0,0],
                               function=Logistic(gain=0.1, bias=0.2),
                               noise=5,
                               time_constant = 0.1,
                               time_scale=TimeScale.TIME_STEP
                               )
        my_Transfer_Test.execute([1,2])
    except FunctionError:
        print("Passed")
        print("")

    print("-------------------------------------------------")

    print("Transfer Test #2: Execute Transfer with noise = 5.0")

    my_Transfer_Test2 = TransferMechanism(name='my_Transfer_Test2',
                            default_input_value = [0,0,0,0],
                            function=Logistic(gain=0.1, bias=0.2),
                            noise=NormalDist().function,
                            time_constant = 0.1,
                            time_scale=TimeScale.TIME_STEP
                            )
    my_Transfer_Test2.execute([1,1,1,1])

    print("Passed")
    print("")

    print("-------------------------------------------------")

    print("Transfer Test #3: Execute Transfer with noise not specified")

    my_Transfer_Test3 = TransferMechanism(name='my_Transfer_Test3',
                            default_input_value = [0,0,0,0,0],
                            function=Logistic(gain=0.1, bias=0.2),
                            time_constant = 0.2,
                            time_scale=TimeScale.TIME_STEP
                            )
    my_Transfer_Test3.execute([10,20,30,40,50])

    print("Passed")
    print("")

    print("-------------------------------------------------")

if run_distribution_test:
    print("Distribution Test #1: Execute Transfer with noise = WaldDist(scale = 2.0, mean = 2.0).function")


    my_Transfer = TransferMechanism(name='my_Transfer',
                           default_input_value = [0,0],
                           function=Logistic(gain=0.1, bias=0.2),
                           noise=WaldDist(scale = 2.0, mean = 2.0).function,
                           time_constant = 0.1,
                           time_scale=TimeScale.TIME_STEP
                           )
    my_Transfer.execute([1,1])

    print("Passed")
    print("")

    print("-------------------------------------------------")

    print("Distribution Test #2: Execute Transfer with noise = GammaDist(scale = 1.0, shape = 1.0).function")


    my_Transfer2 = TransferMechanism(name='my_Transfer2',
                           default_input_value = [0,0],
                           function=Logistic(gain=0.1, bias=0.2),
                           noise=GammaDist(scale = 1.0, shape = 1.0).function,
                           time_constant = 0.1,
                           time_scale=TimeScale.TIME_STEP
                           )
    my_Transfer2.execute([1,1])

    print("Passed")
    print("")

    print("-------------------------------------------------")

    print("Distribution Test #3: Execute Transfer with noise = UniformDist(low = 2.0, high = 3.0).function")

    my_Transfer3 = TransferMechanism(name='my_Transfer3',
                           default_input_value = [0,0],
                           function=Logistic(gain=0.1, bias=0.2),
                           noise=UniformDist(low = 2.0, high = 3.0).function,
                           time_constant = 0.1,
                           time_scale=TimeScale.TIME_STEP
                           )
    my_Transfer3.execute([1,1])

    print("Passed")
    print("")

    print("-------------------------------------------------")

    print("Distribution Test #4: Execute Transfer with noise = ExponentialDist(beta=1.0).function")

    my_Transfer4 = TransferMechanism(name='my_Transfer4',
                           default_input_value = [0,0],
                           function=Logistic(gain=0.1, bias=0.2),
                           noise=ExponentialDist(beta=1.0).function,
                           time_constant = 0.1,
                           time_scale=TimeScale.TIME_STEP
                           )
    my_Transfer4.execute([1,1])

    print("Passed")
    print("")

    print("-------------------------------------------------")

    print("Distribution Test #5: Execute Transfer with noise = NormalDist(mean=1.0, standard_dev = 2.0).function")

    my_Transfer5 = TransferMechanism(name='my_Transfer5',
                           default_input_value = [0,0],
                           function=Logistic(gain=0.1, bias=0.2),
                           noise=NormalDist(mean=1.0, standard_dev = 2.0).function,
                           time_constant = 0.1,
                           time_scale=TimeScale.TIME_STEP
                           )
    my_Transfer5.execute([1,1])

    print("Passed")
    print("")

    print("-------------------------------------------------")

