from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Globals.Keywords import *
from matplotlib import pyplot as plt

threshold = 10

plt.ion()

time = 0
position = 0

axes = plt.gca()
axes.set_ylim([-threshold -5, threshold +5])
plt.axhline(y=threshold, linewidth=1, color = 'k', linestyle= 'dashed')
plt.axhline(y=-threshold, linewidth=1, color = 'k', linestyle= 'dashed')
plt.plot()


my_DDM = DDM(function = DDMIntegrator(drift_rate = 0.05 , noise = 0.1),
             name='My_DDM',
             time_scale = TimeScale.TIME_STEP
             )
while abs(position) < threshold:
    time += 1
    position = my_DDM.execute()[0][0]
    plt.plot(time, position, '-o', color='r', ms=4)
    plt.pause(0.1)

plt.pause(5)

