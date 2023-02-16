
__all__ = ['PNLWarning', 'PNLInternalWarning', 'PNLUserWarning']

class PNLWarning(Warning):
    pass

class PNLInternalWarning(PNLWarning):
    pass

class PNLUserWarning(PNLWarning):
    pass
