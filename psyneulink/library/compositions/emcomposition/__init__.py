from . import emcomposition

from .emcomposition import *
from .emcomposition2 import *
__all__ = list(emcomposition.__all__)
__all__.extend(emcomposition2.__all__)

try:
    import torch
    from .pytorchEMwrappers import *
    __all__.extend(pytorchEMcompositionwrapper.__all__)
except:
    pass
