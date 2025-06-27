from . import emcomposition

from .emcomposition import *
__all__ = list(emcomposition.__all__)

try:
    import torch
    from .pytorchEMwrappers import *
    __all__.extend(pytorchEMcompositionwrapper.__all__)
except:
    pass
