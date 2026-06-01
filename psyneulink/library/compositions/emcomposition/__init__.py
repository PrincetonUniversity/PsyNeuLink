from . import emcomposition_proj

from .emcomposition_proj import *
from .emcomposition import *
__all__ = list(emcomposition.__all__)
__all__.extend(emcomposition_proj.__all__)

try:
    import torch
    from .pytorchEMwrappersProj import *
    __all__.extend(pytorchEMcompositionwrapper.__all__)
except:
    pass
