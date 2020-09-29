# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# *********************************************** KohoneMechanism ******************************************************

"""

Contents
--------

  * `KohonenMechanism_Overview`
  * `KohonenMechanism_Creation`
  * `KohonenMechanism_Structure`
      - `KohonenMechanism_Learning`
  * `KohonenMechanism_Execution`
  * `KohonenMechanism_Class_Reference`


.. _KohonenMechanism_Overview:

Overview
--------

A KohonenMechanism is a subclass of `RecurrentTransferMechanism` that implements a `Kohonen network
<http://www.scholarpedia.org/article/Kohonen_network>`_ (`brief explanation
<https://www.cs.bham.ac.uk/~jlw/sem2a2/Web/Kohonen.htm>`_; `nice demo <https://www.youtube.com/watch?v=QvI6L-KqsT4>`_),
which is a particular form of `self-organized map (SOM) <https://en.wikipedia.org/wiki/Self-organizing_map>`_. By
default, a KohonenMechanism uses a `KohonenLearningMechanism` and the `Kohonen` `LearningFunction <LearningFunctions>`
to implement implement a form of unsupervised learning that produces the self-organized map.

.. _KohonenMechanism_Creation:

Creating a KohonenMechanism
---------------------------

A KohonenMechanism can be created directly by calling its constructor.

.. _KohonenMechanism_Structure:

Structure
---------

TBD

.. _KohonenMechanism_Learning:

Learning
~~~~~~~~

TBD

.. _KohonenMechanism_Execution:

Execution
---------

TBD

.. _KohonenMechanism_Class_Reference:

Class Reference
---------------

"""

import itertools
import logging
import numbers
import warnings

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.learningfunctions import Kohonen
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    ACTIVATION_INPUT, ACTIVATION_OUTPUT, LearningMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    DEFAULT_MATRIX, FUNCTION, GAUSSIAN, IDENTITY_MATRIX, INITIALIZING, KOHONEN_MECHANISM, \
    LEARNED_PROJECTIONS, LEARNING_SIGNAL, MATRIX, MAX_INDICATOR, NAME, OWNER_VALUE, OWNER_VARIABLE, RESULT, VARIABLE
from psyneulink.core.globals.parameters import Parameter, SharedParameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.library.components.mechanisms.modulatory.learning.kohonenlearningmechanism import KohonenLearningMechanism

__all__ = [
    'INPUT_PATTERN', 'KohonenMechanism', 'KohonenError', 'MAXIMUM_ACTIVITY'
]

logger = logging.getLogger(__name__)


class KohonenError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


MAXIMUM_ACTIVITY = 'MAXIMUM_ACTIVITY'
INPUT_PATTERN = 'INPUT_PATTERN'


class KohonenMechanism(TransferMechanism):
    """
    KohonenMechanism(                                          \
        enable_learning=True,                                  \
        learning_function=Kohonen(distance_function=GAUSSIAN), \
        learning_rate=None)

    Subclass of `TransferMechanism` that learns a `self-organized <https://en.wikipedia.org/wiki/Self-organizing_map>`_
    map of its input.
    See `TransferMechanism <TransferMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    COMMENT:
    selection_function : SelectionFunction, function or method : default OneHot(mode=MAX_VAL)
        specifes the function used to select the element of the input used to train the `matrix
        <MappingProjection.matrix>` of afferent `MappingProjection` to the Mechanism.
    COMMENT

    enable_learning : boolean : default True
        specifies whether the Mechanism should be configured for learning;  if it is not (the default), then learning
        cannot be enabled until it is configured for learning by calling the Mechanism's `configure_learning
        <KohonenMechanism.configure_learning>` method.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <KohonenMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <KohonenMechanism.learning_rate>` for details).

    learning_function : LearningFunction, function or method : default Kohonen(distance_function=GUASSIAN)
        specifies function used by `learning_mechanism <KohonenMechanism.learning_mechanism>` to update `matrix
        <MappingProjection.matrix>` of `learned_projection <KohonenMechanism.learned_projection>.


    Attributes
    ----------

    COMMENT:
    selection_function : SelectionFunction, function or method : default OneHot(mode=MAX_VAL)
        determines the function used to select the element of the input used to train the `matrix
        <MappingProjection.matrix>` of the `learned_projection <KohonenMechanism.learned_projection>`.
    COMMENT

    distance_function : Function, function or method : default Gaussian
        determines the function used to evaluate the distance of each element from the most active one
        one identified by `selection_function <KohonenMechanism.selection_function>`

    matrix : 2d np.array
        `matrix <AutoAssociativeProjection.matrix>` parameter of the `learned_projection
        <KohonenMechanism.learned_projection>`.

    learning_enabled : bool
        indicates whether `learning is enabled <KohonenMechanism_Learning>`;  see `learning_enabled
        <KohonenMechanism.learning_enabled>` for additional details.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        determines the learning rate used by the `learning_function <KohonenMechanism.learning_function>`
        of the `learning_mechanism <KohonenMechanism.learning_mechanism>` (see `learning_rate
        <KohonenLearningMechanism.learning_rate>` for details concerning specification and default value assignment).

    learned_projection : MappingProjection
        `MappingProjection` that projects to the Mechanism and is trained by its `learning_mechanism
        <KohonenMechanism.learning_mechanism>`.

    learning_function : LearningFunction, function or method
        function used by `learning_mechanism <KohonenMechanism.learning_mechanism>` to update `matrix
        <MappingProjection.matrix>` of `learned_projection <KohonenMechanism.learned_projection>.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <KohonenMechanism_Learning>`, and used to train the
        `learned_projection <KohonenMechanism.learned_projection>`.

    output_ports : Dict[str, OutputPort]
        an OrderedDict with the following `OutputPorts <OutputPort>`:

        * *RESULT* -- `value <OutputPort.value>` is the result of `function <Mechanism_Base.function>`;

        * *INPUT_PATTERN* -- `value <OutputPort.value>` is the value of the KohonenMechanism's `variable
          <Mechanisms_Base.variable>`, which is provided to the *ACTIVATION_OUTPUT* InputPort of its
          `learning_mechanism <KohonenMechanisms.learning_mechanism>`.

    standard_output_ports : list[dict]
        list of `Standard OutputPort <OutputPort_Standard>` that includes the following in addition to
        the `standard_output_ports <TransferMechanism.standard_output_ports>` of a `TransferMechanism`:

        .. _MAXIMUM_ACTIVITY:

        *MAXIMUM_ACTIVITY* : 1d array
             "one hot" encoding of the most active element of the Mechanism's `value <Mechanism_Base.value>` in
             the last execution.


    Returns
    -------
    instance of KohonenMechanism : KohonenMechanism

    """

    componentType = KOHONEN_MECHANISM

    class Parameters(TransferMechanism.Parameters):
        """
            Attributes
            ----------

                enable_learning
                    see `enable_learning <KohonenMechanism.enable_learning>`

                    :default value: True
                    :type: ``bool``

                learning_function
                    see `learning_function <KohonenMechanism.learning_function>`

                    :default value: `Kohonen`
                    :type: `Function`

                learning_rate
                    see `learning_rate <KohonenMechanism.learning_rate>`

                    :default value: None
                    :type:

                matrix
                    see `matrix <KohonenMechanism.matrix>`

                    :default value: `AUTO_ASSIGN_MATRIX`
                    :type: ``str``

                output_ports
                    see `output_ports <KohonenMechanism.output_ports>`

                    :default value: [`RESULT`, "{name: INPUT_PATTERN, variable: OWNER_VARIABLE}"]
                    :type: ``list``
                    :read only: True
        """
        learning_function = SharedParameter(
            Kohonen(distance_function=GAUSSIAN),
            attribute_name='learning_mechanism',
            shared_parameter_name='function',
        )
        enable_learning = True
        matrix = DEFAULT_MATRIX

        output_ports = Parameter(
            [RESULT, {NAME: INPUT_PATTERN, VARIABLE: OWNER_VARIABLE}],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

    standard_output_ports = TransferMechanism.standard_output_ports.copy()
    standard_output_ports.extend([{NAME:MAXIMUM_ACTIVITY,
                                    VARIABLE:(OWNER_VALUE,0),
                                    FUNCTION: OneHot(mode=MAX_INDICATOR)}
                                   ])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=None,
                 # selection_function=OneHot(mode=MAX_INDICATOR),  # RE-INSTATE WHEN IMPLEMENT NHot function
                 integrator_function=None,
                 initial_value=None,
                 noise: tc.optional(is_numeric_or_none) = None,
                 integration_rate: tc.optional(is_numeric_or_none) = None,
                 integrator_mode=None,
                 clip=None,
                 enable_learning=None,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.optional(is_function_type) = None,
                 learned_projection:tc.optional(MappingProjection)=None,
                 additional_output_ports:tc.optional(tc.any(str, Iterable))=None,
                 name=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs
                 ):
        # # Default output_ports is specified in constructor as a string rather than a list
        # # to avoid "gotcha" associated with mutable default arguments
        # # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        # if output_ports is None:
        #     output_ports = [RESULT]

        output_ports = [RESULT, {NAME: INPUT_PATTERN, VARIABLE: OWNER_VARIABLE}]
        if additional_output_ports:
            if isinstance(additional_output_ports, list):
                output_ports += additional_output_ports
            else:
                output_ports.append(additional_output_ports)

        self._learning_enabled = enable_learning
        self._learning_enable_deferred = False

        super().__init__(
            default_variable=default_variable,
            size=size,
            function=function,
            integrator_function=integrator_function,
            integrator_mode=integrator_mode,
            learning_rate=learning_rate,
            learning_function=learning_function,
            learned_projection=learned_projection,
            enable_learning=enable_learning,
            initial_value=initial_value,
            noise=noise,
            integration_rate=integration_rate,
            clip=clip,
            output_ports=output_ports,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set, target_set, context)

        if LEARNED_PROJECTION in target_set and target_set[LEARNED_PROJECTION]:
            if target_set[LEARNED_PROJECTION].receiver.owner != self:
                raise KohonenError("{} specified in {} argument for {} projects to a different Mechanism ({})".
                                   format(MappingProjection.__name__,
                                          repr(LEARNED_PROJECTION), self.name,
                                          target_set[LEARNED_PROJECTION].receiver.owner.name))

    def _instantiate_attributes_after_function(self, context=None):
        super()._instantiate_attributes_after_function(context)
        if self.learning_enabled:
            self.configure_learning(context=context)

    # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
    @handle_external_context()
    def configure_learning(self,
                           learning_function:tc.optional(tc.any(is_function_type))=None,
                           learning_rate:tc.optional(tc.any(numbers.Number, list, np.ndarray, np.matrix))=None,
                           learned_projection:tc.optional(MappingProjection)=None,
                           context=None):
        """Provide user-accessible-interface to _instantiate_learning_mechanism

        Configure KohonenMechanism for learning. Creates the following Components:

        * a `LearningMechanism` -- if the **learning_function** and/or **learning_rate** arguments are
          specified, they are used to construct the LearningMechanism, otherwise the values specified in the
          KohonenMechanism's constructor are used;
        ..
        * a `MappingProjection` from the KohonenMechanism's `primary OutputPort <OutputPort_Primary>`
          to the LearningMechanism's *ACTIVATION_INPUT* InputPort;
        ..
        * a `LearningProjection` from the LearningMechanism's *LEARNING_SIGNAL* OutputPort to the learned_projection;
          by default this is the KohonenMechanism's `learned_projection <KohonenMechanism.learned_projection>`;
          however a different one can be specified.

        """
        # This insures that these are validated if the method is called from the command line (i.e., by the user)
        if learning_function:
            self.learning_function = learning_function
        if learning_rate:
            self.learning_rate = learning_rate
        if learned_projection:
            self.learned_projection = learned_projection

        # Assign learned_projection, using as default the first Projection to the Mechanism's primary InputPort
        try:
            self.learned_projection = self.learned_projection or self.input_port.path_afferents[0]
        except:
            self.learned_projection = None
        if not self.learned_projection:
            # Mechanism already belongs to a Process or System, so should have a MappingProjection by now
            if (self.processes or self.systems):
                raise KohonenError("Configuring learning for {} requires that it receive a {} "
                                   "from another {} within a {} to which it belongs".
                                   format(self.name, MappingProjection.__name__, Mechanism.__name__, Process.__name__))
                                   # "receive at least one {} or that the {} be specified".
                                   # format(self.name, MappingProjection.__name__, repr(LEARNED_PROJECTION)))
            # Mechanism doesn't yet belong to a Process or System, so wait until then to configure learning
            #  (this method will be called again from _add_projection_to_mechanism if a Projection is added)
            else:
                self._learning_enable_deferred = True
                return

        self.parameters.matrix._set(self.learned_projection.parameter_ports[MATRIX], context)

        self.learning_mechanism = self._instantiate_learning_mechanism(learning_function=self.learning_function,
                                                                       learning_rate=self.learning_rate,
                                                                       learned_projection=self.learned_projection,
                                                                       context=context)

        self.learning_projection = self.learning_mechanism.output_ports[LEARNING_SIGNAL].efferents[0]

        if self.learning_mechanism is None:
            self.learning_enabled = False

    # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
    def _instantiate_learning_mechanism(self,
                                        learning_function,
                                        learning_rate,
                                        learned_projection,
                                        context=None):

        learning_mechanism = KohonenLearningMechanism(default_variable=[self.learned_projection.sender.value,
                                                                        self.learned_projection.receiver.value],
                                                      matrix=self.matrix,
                                                      function=learning_function,
                                                      learning_rate=learning_rate,
                                                      # learning_signals=[self.matrix],
                                                      name="{} for {}".format(
                                                              LearningMechanism.className,
                                                              self.name))

        # KDM 10/22/18: should below be aux_components?
        # FIX: 10/31/19 [JDC]: YES!

        # Instantiate Projection from learned_projection's sender to LearningMechanism
        MappingProjection(sender=self.learned_projection.sender,
                          receiver=learning_mechanism.input_ports[ACTIVATION_INPUT],
                          matrix=IDENTITY_MATRIX,
                          name="Error Projection for {}".format(learning_mechanism.name))

        # Instantiate Projection from the Mechanism's INPUT_PATTERN OutputPort
        #    (which has the value of the learned_projection's receiver;  i.e., the Mechanism's input)
        #    to the LearningMechanism's ACTIVATION_OUTPUT InputPort.
        MappingProjection(sender=self.output_ports[INPUT_PATTERN],
                          receiver=learning_mechanism.input_ports[ACTIVATION_OUTPUT],
                          matrix=IDENTITY_MATRIX,
                          name="Error Projection for {}".format(learning_mechanism.name))

        # Instantiate Projection from LearningMechanism to learned_projection
        LearningProjection(sender=learning_mechanism.output_ports[LEARNING_SIGNAL],
                           receiver=self.matrix,
                           name="{} for {}".format(LearningProjection.className, self.learned_projection.name))

        return learning_mechanism

    def _projection_added(self, projection, context=None):
        super()._projection_added(projection, context)
        if self._learning_enable_deferred:
            self.configure_learning(context=context)

    @property
    def learning_enabled(self):
        return self._learning_enabled

    @learning_enabled.setter
    def learning_enabled(self, value:bool):

        self._learning_enabled = value
        # Enable learning for KohonenMechanism's learning_mechanism
        if self.learning_mechanism is not None:
            self.learning_mechanism.learning_enabled = value
        # If KohonenMechanism has no LearningMechanism, warn and then ignore attempt to set learning_enabled
        elif value is True:
            warnings.warn("Learning cannot be enabled for {} because it has no {}".
                  format(self.name, LearningMechanism.__name__))
            return

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            [self.learning_mechanism],
            [self.learning_projection],
        ))
