from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Functions.Process import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *

myMechanism = DDM(function=NavarroAndFuss(drift_rate=1.0,
                                      threshold=10.0,
                                      starting_point=0.0),
                  name='My_DDM',
                  prefs={REPORT_OPUTPUT_PREF: PreferenceEntry(True,PreferenceLevel.INSTANCE)}
                  )

# simple_ddm_process = process('Simple DDM Process')
simple_ddm_process = process(configuration=[myMechanism],
                             prefs={REPORT_OPUTPUT_PREF: True})
simple_ddm_process.execute(1.0)

