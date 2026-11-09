# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


"""Hierarchical parameter estimation for `ParameterEstimationComposition`.

Fits a group of participants jointly rather than one at a time, so that each participant's estimate
is informed by the population they belong to.

`HierarchicalPECResults` and `HierarchicalEMWarning` are the public surface.
"""

# ParameterEstimationComposition imports this package, so nothing here may import it back at
# module scope; use a function-local import.
from . import hierarchicalresults, laplaceem, subjectlikelihood, transforms
from .hierarchicalresults import HierarchicalPECResults
from .laplaceem import HierarchicalEMWarning

__all__ = ["HierarchicalEMWarning", "HierarchicalPECResults"]
