from Functions.Mechanisms.DDM import *
from Functions.Process import Process

# x = Mechanism()
# x = Mechanism('DDM')
# x = DDM('Jon')
# x = DDM()
# x = mechanism("DDM")

input = .1
DDM_allocations = {kwDriftRate: 1,
                   kwStartingPoint: 1,
                   kwThreshold: 1,
                   kwT0: 1,
                   kwNoise: 1
                   }

allocations = {'DDM-1':DDM_allocations}

# x = mechanism()
y = Process()

driftRateParam = {kwDriftRate: ParamControlTuple(param=DDMParamValuesTuple(value=0,
                                                                           variability=0,
                                                                           distribution=NotImplemented),
                                            controlSignal={
                                                kwControlSignalIdentity: [1]
                                                # kwControlSignalSettings: NotImplemented,
                                                # kwControlSignalAllocationSamplingRange:NotImplemented,
                                                # kwControlSignalLogProfile: NotImplemented
                                            })}

# x.adjust_function(driftRateParam)
# x.execute_function(input, allocations, TimeScale.TRIAL)
# x.execute_function(NotImplemented, [1,2], TimeScale.TRIAL,NotImplemented)

# y.adjust(driftRateParam)
test_input = [.1]
y.execute(input=test_input,
          mechanism_control_allocations_dict=allocations,
          time_scale=TimeScale.TRIAL)

print("\nMechanismsRegistry:\n", MechanismRegistry)
print("\n\nget_mechanims:\n",get_mechanisms())
print("\n\nProcess:\n",y.get_mechanism_dict())
