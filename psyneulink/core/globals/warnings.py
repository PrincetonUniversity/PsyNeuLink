import warnings

__all__ = ['PNLWarning', 'PNLInternalWarning', 'PNLUserWarning']

class PNLWarning(Warning):
    pass

class PNLInternalWarning(PNLWarning):
    pass

class PNLUserWarning(PNLWarning):
    pass


def _disable_internal_warnings():
    warnings.simplefilter("ignore", PNLInternalWarning)

def _enable_internal_warnings():
    warnings.simplefilter("default", PNLInternalWarning)


# Disable internal warnings by default
_disable_internal_warnings()
