from . import composition
from . import systemcomposition
from . import pathwaycomposition
from . import autodiffcomposition

from .composition import *
from .systemcomposition import *
from .pathwaycomposition import *

__all__ = list(composition.__all__)
__all__.extend(systemcomposition.__all__)
__all__.extend(pathwaycomposition.__all__)

try:
    import torch
    from torch import nn
    torch_available = True
except ImportError:
    torch_available = False

if torch_available:
    from .autodiffcomposition import *
    __all__.extend(autodiffcomposition.__all__)