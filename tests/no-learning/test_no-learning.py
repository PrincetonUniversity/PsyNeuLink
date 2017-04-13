import logging

from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Functions.Function import Linear, Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import REPORT_OUTPUT_PREF, VERBOSE_PREF
from PsyNeuLink.Globals.Keywords import FULL_CONNECTIVITY_MATRIX, LEARNING_PROJECTION
from PsyNeuLink.Globals.TimeScale import TimeScale
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import AfterNCalls

logger = logging.getLogger(__name__)

class TestNoLearning:
    def test_stroop(self):
        process_prefs = {REPORT_OUTPUT_PREF: True,
                     VERBOSE_PREF: False}

        system_prefs = {REPORT_OUTPUT_PREF: False,
                        VERBOSE_PREF: False}

        colors = TransferMechanism(default_input_value=[0,0],
                                function=Linear,
                                name="Colors")

        words = TransferMechanism(default_input_value=[0,0],
                                function=Linear,
                                name="Words")

        response = TransferMechanism(default_input_value=[0,0],
                                   function=Logistic,
                                   name="Response")

        color_naming_process = process(
            default_input_value=[1, 2.5],
            # pathway=[(colors, 0), FULL_CONNECTIVITY_MATRIX, (response,0)],
            pathway=[colors, FULL_CONNECTIVITY_MATRIX, response],
            learning=LEARNING_PROJECTION,
            target=[1,2],
            name='Color Naming',
            prefs=process_prefs
        )

        word_reading_process = process(
            default_input_value=[.5, 3],
            pathway=[words, FULL_CONNECTIVITY_MATRIX, response],
            name='Word Reading',
            learning=LEARNING_PROJECTION,
            target=[3,4],
            prefs=process_prefs
        )

        mySystem = system(processes=[color_naming_process, word_reading_process],
                          name='Stroop Model',
                          targets=[0,0],
                          prefs=system_prefs,
                          )

        sched = Scheduler(system=mySystem)
        mySystem.scheduler = sched

        term_conds = {TimeScale.TRIAL: AfterNCalls(response, 2)}
        mySystem.execute(termination_conditions=term_conds)