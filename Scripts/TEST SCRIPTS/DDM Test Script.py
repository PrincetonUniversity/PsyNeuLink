from PsyNeuLink.Components.Process import process
from PsyNeuLink.Globals.Keywords import *

DDM_prefs = ComponentPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

# my_DDM = DDM(name='my_DDM')

my_DDM = DDM(function=BogaczEtAl(drift_rate=(1.0, CONTROL_PROJECTION),
                                 threshold=(10.0, CONTROL_PROJECTION),
                                 starting_point=0.0),
             name='My_DDM',
             prefs = DDM_prefs
             )

my_DDM.prefs = DDM_prefs
my_DDM.prefs.level = PreferenceLevel.SYSTEM

my_DDM.prefs.show()

myMechanism = DDM(function=BogaczEtAl(drift_rate=(1.0, CONTROL_PROJECTION),
                                      threshold=(10.0, CONTROL_PROJECTION),
                                      starting_point=0.0),
                  prefs = DDM_prefs,
                  name='My_DDM'
                  )

myMechanism_2 = DDM(function=BogaczEtAl(drift_rate=2.0,
                                        threshold=20.0),
                    prefs = DDM_prefs,
                    name='My_DDM_2'
                    )

myMechanism_3 = DDM(function=BogaczEtAl(drift_rate=3.0,
                                        threshold=30.0),
                    prefs = DDM_prefs,
                    name='My_DDM_3'
                    )

process_prefs = ComponentPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

process_prefs.show()

z = process(default_variable=[[30], [10]],
            params={PATHWAY:[myMechanism,
                                   (IDENTITY_MATRIX),
                                   myMechanism_2,
                                   (FULL_CONNECTIVITY_MATRIX),
                                   myMechanism_3]},
            prefs = process_prefs)

z.execute([[30], [10]])

myMechanism.log.print_entries(ALL_ENTRIES, kwTime, kwValue)

