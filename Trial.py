class Trial:
    def __init__(self,
                 name="Trial",
                 number=None,
                 stimuli=[]
                 ):
        """Initializes the default structure of a trial

            :parameter name: (default: "Trial")
            :parameter number: number in sequence (default: 0)
            :parameter stimuli: list of Stimulus objects to present (default: empty list)
            """

class Stimulus:
    def __init__(self,
                 name="Stimulus",
                 number=None,
                 stimulus=[],
                 onset_time=0,
                 duration=0
                 ):
        """Initializes the default structure of a stimulu

            :parameter name: (default: "Stimulus")
            :parameter number: number in sequence (default: 0)
            :parameter stimulus: object to present (default: empty list)
            :parameter onset_time: clock time at which to present this stimulus (default: 0)
            :parameter duration: duration to present stimulus;  it is cleared after this duration (default: 0)
        """



