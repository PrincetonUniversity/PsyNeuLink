"""Built-in batched op specs.

Importing this package registers the batched op specs for the supported
PsyNeuLink components.  Each module holds one component's complete batched
definition (declaration, CPU body, Triton body) and doubles as a reference
example for registering new components.
"""

from psyneulink.core.batched.components import (  # noqa: F401
    ddm,
    lca,
    linear,
    logistic,
    mapping_projection,
    passthrough,
)
