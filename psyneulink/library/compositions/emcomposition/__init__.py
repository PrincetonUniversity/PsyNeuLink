from . import emcomposition

from .emcomposition import *
from .emcomposition2 import *
__all__ = list(emcomposition2.__all__)
__all__.extend(emcomposition.__all__)

try:
    import torch
    from .pytorchEMwrappers import *
    __all__.extend(pytorchEMcompositionwrapper.__all__)
except:
    pass
