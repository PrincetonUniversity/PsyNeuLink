# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  AutoAssociativeProjection ***********************************************

"""
.. _Auto_Associative_Overview:

Intro
-----

I am basically just a MappingProjection, except I'm intended to be used as a recurrent projection. I thus require
that my matrix (with which I multiply my input to produce my output) is a square matrix. By default I point to/from the
primary input state and output state of my owner. But you can specify the input and output state as well.
"""

from PsyNeuLink.Components.Projections.Projection import *
from PsyNeuLink.Components.Projections.PathwayProjections.PathwayProjection import PathwayProjection_Base
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Functions.Function import *

parameter_keywords.update({AUTO_ASSOCIATIVE_PROJECTION})
projection_keywords.update({AUTO_ASSOCIATIVE_PROJECTION})

class AutoAssociativeError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

class AutoAssociativeProjection(MappingProjection):
    """
    Insert docs here
    """

    componentType = AUTO_ASSOCIATIVE_PROJECTION
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # necessary?
    paramClassDefaults = MappingProjection.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 sender=None,
                 receiver=None,
                 matrix=DEFAULT_MATRIX,
                 auto=None,
                 cross=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=None):

        if owner is not None:
            if not isinstance(owner, Mechanism):
                raise AutoAssociativeError('Owner of AutoAssociative Mechanism must either be None or Mechanism')
            if sender is None:
                sender = owner
            if receiver is None:
                receiver = owner

        self._assign_args_to_param_dicts(auto=auto, cross=cross, params=params)

        super().__init__(sender=sender,
                         receiver=receiver,
                         matrix=matrix,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate recurrent_projection, matrix, and the functions for the ENERGY and ENTROPY outputStates
        """

        super()._instantiate_attributes_after_function(context=context)
        print("self.params: ", self.params)
        auto = self.params[AUTO]
        cross = self.params[CROSS]
        if auto is not None and cross is not None:
            a = get_matrix(IDENTITY_MATRIX, self.size[0], self.size[0]) * auto
            c = get_matrix(HOLLOW_MATRIX, self.size[0], self.size[0]) * cross
            self.matrix = a + c
        if auto is not None:
            self.matrix = get_matrix(IDENTITY_MATRIX, self.size[0], self.size[0]) * auto
        if cross is not None:
            self.matrix = get_matrix(HOLLOW_MATRIX, self.size[0], self.size[0]) * cross