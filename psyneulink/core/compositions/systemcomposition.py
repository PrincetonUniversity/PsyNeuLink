from psyneulink.core.compositions.composition import Composition

__all__ = [
    'SystemComposition', 'SystemCompositionError'
]


class SystemCompositionError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class SystemComposition(Composition):
    """

            Arguments
            ----------

            Attributes
            ----------

            Returns
            ----------
    """

    def __init__(self):
        super(SystemComposition, self).__init__()

