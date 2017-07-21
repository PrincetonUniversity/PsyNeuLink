from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Components.Process import process



import time
print("========TIME STEP MODE========")
print()
print("MECHANISM ")
my_DDM = DDM(function=Integrator(integration_type=DIFFUSION, noise=0.5),
             name='My_DDM',
             time_scale=TimeScale.TIME_STEP
             )
print("result  = ", my_DDM.execute(500))
print("--------------")
time.sleep(0.3)
print("PROCESS ")
my_Transfer_Test = TransferMechanism(name='my_Transfer_Test',
                        function=Linear()
                        )
my_DDM_proc = DDM(function=Integrator(integration_type=DIFFUSION, noise=0.5),
             name='My_DDM_Proc',
             time_scale=TimeScale.TIME_STEP
             )

MyProcess = process(
    pathway=[my_Transfer_Test, my_DDM_proc],
    name='MyProcess')
print("RESULT = ", MyProcess.execute(500))
print("--------------")
print()
print()
time.sleep(0.3)


print("========TRIAL MODE========")


print()
print("MECHANISM ")
my_DDM_Trial = DDM()
print("RESULT  = ", my_DDM_Trial.execute(500))
print("--------------")
time.sleep(0.3)
print("PROCESS ")
my_Transfer_Test_Trial = TransferMechanism(name='my_Transfer_Test_Trial',
                        function=Linear()
                        )
my_DDM_proc_trial = DDM()

MyProcess2 = process(
    pathway=[my_Transfer_Test_Trial, my_DDM_proc_trial],
    name='MyProcess')

print("RESULT = ", MyProcess2.execute([500]))
print("--------------")
time.sleep(0.3)