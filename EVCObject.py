from Functions.Projections import ControlSignal

# import Process

CONTROL_SIGNAL_COLOR = 0
PROCESS_COLOR = 1


class EVCObject:
    """General class for all EVC objects

    Instance attributes:
        + color — for use in interface design
        + activity - for use in modeling activity (e.g., fMRI, EEG, etc.);  function assigned by child class
    """

    def __init__(self):

        # Assign colors to EVC objects
        if isinstance(self, ControlSignal):
            self.color = CONTROL_SIGNAL_COLOR
        # elif isinstance(self, Process.Process):
        #     self.color = PROCESS_COLOR

        self.activity = NotImplemented
        self.activityFunction = NotImplemented


    # Setters and getters

    def set_activity(self, activity):
        self.activity = activity

    def get_activity(self):
        return self.activity

