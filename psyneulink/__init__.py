# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************

"""
PsyNeuLink is a "block modeling system" for cognitive neuroscience.

Documentation is available at https://princetonuniversity.github.io/PsyNeuLink/

Example scripts are available at https://github.com/PrincetonUniversity/PsyNeuLink/tree/master/Scripts

If you have trouble installing PsyNeuLink, run into any bugs, or have suggestions for development,
please contact psyneulinkhelp@princeton.edu.
"""

import logging as _logging

import numpy as _numpy
import pint as _pint

# pint requires a package-specific unit registry, and to use it as
# needed in psyneulink, it has to come before imports. This is the
# reason for skipping E402 below
_unit_registry = _pint.get_application_registry()
_pint.set_application_registry(_unit_registry)
_unit_registry.precision = 8  # TODO: remove when floating point issues resolved

# starred imports to allow user imports from top level
from . import core  # noqa: E402
from . import library  # noqa: E402

from ._version import get_versions  # noqa: E402
from .core import *  # noqa: E402
from .library import *  # noqa: E402


_pnl_global_names = [
    'primary_registries', 'System', 'Process', '_unit_registry',
]
# flag when run from pytest (see conftest.py)
_called_from_pytest = False

__all__ = list(_pnl_global_names)
__all__.extend(core.__all__)
__all__.extend(library.__all__)

# set __version__ based on versioneer
__version__ = get_versions()['version']
del get_versions

# suppress numpy overflow and underflow errors
_numpy.seterr(over='ignore', under='ignore')


# https://stackoverflow.com/a/17276457/3131666
class _Whitelist(_logging.Filter):
    def __init__(self, *whitelist):
        self.whitelist = [_logging.Filter(name) for name in whitelist]

    def filter(self, record):
        return any(f.filter(record) for f in self.whitelist)


class _Blacklist(_Whitelist):
    def filter(self, record):
        return not _Whitelist.filter(self, record)


_logging.basicConfig(
    level=_logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for handler in _logging.root.handlers:
    handler.addFilter(_Blacklist(
        'psyneulink.core.scheduling.scheduler',
        'psyneulink.core.scheduling.condition',
    ))

primary_registries = [
    CompositionRegistry,
    ControlMechanismRegistry,
    DeferredInitRegistry,
    FunctionRegistry,
    GatingMechanismRegistry,
    MechanismRegistry,
    PathwayRegistry,
    PortRegistry,
    PreferenceSetRegistry,
    ProjectionRegistry,
]

for reg in primary_registries:
    def func(name, obj):
        if isinstance(obj, Component):
            obj._is_pnl_inherent = True

    process_registry_object_instances(reg, func)

def System(*args, **kwars):
    show_warning_sys_and_proc_warning()

def Process(*args, **kwars):
    show_warning_sys_and_proc_warning()

def show_warning_sys_and_proc_warning():
    raise ComponentError(f"'System' and 'Process' are no longer supported in PsyNeuLink; "
                         f"use 'Composition' and/or 'Pathway' instead")
