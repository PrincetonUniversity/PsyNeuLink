# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* show_graph *************************************************************

"""

.. _ShowGraph_show_graph_Method:

*Use of the show_graph Method*
------------------------------

Every `Composition` has a `show_graph <ShowGraph.show_graph>` method that can be used to generate a graphical display
of the Composition and, optionally, any `nested Compositions <Composition_Nested>` within it.  Each `Node
<Composition_Nodes>` of the Composition is represented as a node in the graph, and `Projections <Projection>` between
them as edges.

.. technical_note::
    Every Composition is assigned a `ShowGraph` object, that is implemented in the free-standing showgraph.py module.
    The `show_graph <ShowGraph.show_graph>` method of a Composition directly calls the `show_graph
    <ShowGraph.show_graph>` method of its `ShowGraph` object, as do all links to documentation concerning
    `show_graph`.

By default, all nodes within a Composition, including any `Compositions nested <Composition_Nested>` within it, are
shown, each displayed as an oval (if the node is a `Mechanism`) or a rectangle (if it is a nested Composition),
and labeled by its `name <Registry_Naming>`.  Each Composition's `INPUT` Nodes are shown in green, its `OUTPUT`
Nodes are shown in red, and any that are both (i.e., are `SINGLETON`\\s) are shown in brown.  Projections shown as
unlabeled arrows, as illustrated for the Composition in the `examples <ShowGraph_Examples_Visualization>`. However,
these and other attributes of the graph can be modified using arguments in the call to the `show_graph
<ShowGraph.show_graph>` method.

*Display structure* -- how much information is displayed for a Composition and any nested within it can be modified
using the *show_xxx* arguments;  for example, **show_node_structure** determines how much detail is shown about each
Node; **show_nested** determines whether nested Compositions are shown embedded within their enclosing Compositions
or as separate insets, and how many levels of nesting to show; **show_controller** determines whether or not to show
a Composition's `controller <Composition_Controller>`;  and **show_learning** determines whether or not to show its
`learning compnents <Composition_Learning_Components>`.  These are listed as the arguments for the show_graph
<ShowGraph.show_graph>` method below.

*Display attributes* -- the colors, shapes and arrow styles used in the display can be modified using the
**show_graph_attributes** argument of a Composition's constructor, as described below.

.. _ShowGraph_Attributes:

*Display Attributes*
--------------------

The default attributes used to display different types of `Components <Component>` and their `roles <NodeRole>`
within a Composition are listed below. These can be customized using the **show_graph_attributes** argument of a
Composition's constructor, in a dict with keys that are any of the parameters listed in `ShowGraph`, and values
that are any supported by `GraphViz <https://www.graphviz.org>`_ for
`shapes <https://www.graphviz.org/doc/info/shapes.html>`_,
`arrow styles <https://www.graphviz.org/doc/info/arrows.html>`_
`colors <https://www.graphviz.org/doc/info/colors.html>`_.

Shapes
~~~~~~

COMMENT:
FIX: MAKE FIGURE THAT HAS ALL THE VARIOUS TYPES USING CORRESPONDING NAMES
Input Node
Singleton Node
Output Node
LearningMechanism
ControlMechanism
Controller
Nested Composition
COMMENT

`Nested Composition <Composition_Nested>`: square

`Mechanism`:
  - default: oval
  - `CYCLE`: doublecircle
  - `FEEDBACK_SENDER`: octagon
  - `CONTROLLER`: doubleoctagon

`Projection`:
  - default: arrow
  - `ControlProjection`: box
    - 'RANDOMIZATION_CONTROL_SIGNAL` : dashed line
  - `MappingProjection` that receives a `LearningProjection` when **show_learning** is True:  diamond

Colors
~~~~~~

Component-types
^^^^^^^^^^^^^^^

- Control-related components:  blue
- Controller-related: purple
- Learning-related components: orange
- Inactive Projection: red
- Active items (when **animate** = True in `run <Composition.run>`): **BOLD**

Nodes
^^^^^
  - `INPUT`: green
  - `OUTPUT`: red
  - `SINGLETON`: brown
  - `CONTROL`: blue
  - `CONTROLLER` : purple
  - `LEARNING` : orange

.. _ShowGraph_Animation:

*Animation*
-----------

An animation can be generated of the execution of a Composition by using the **animate** argument of the Composition's
`run <Composition.run>` method.  The animation show a graphical display of the Composition, with each of its
the Components highlighted in the sequence that they are executed.  The **animate** can be passed a dict containing
any of the options described above to customize the display, as well as several others used to customize the animation
(see **animate** argument under `run <Composition.run>`).

  .. note::
     At present, animation of the Components within a `nested Composition <Composition_Nested>` is not supported;
     the box surrounding the nested Composition is highlighted when it is executed, followed by the next Component(s)
     to execute.

.. _ShowGraph_Examples_Visualization:

*Examples*
----------

.. _Composition_show_graph_basic_figure:

+-----------------------------------------------------------+----------------------------------------------------------+
| >>> from psyneulink import *                              | .. figure:: _static/Composition_show_graph_basic_fig.svg |
| >>> a = ProcessingMechanism(                              |                                                          |
|               name='A',                                   |                                                          |
| ...           input_shapes=3,                                     |                                                          |
| ...           output_ports=[RESULT, MEAN]                 |                                                          |
| ...           )                                           |                                                          |
| >>> b = ProcessingMechanism(                              |                                                          |
| ...           name='B',                                   |                                                          |
| ...           input_shapes=5                                      |                                                          |
| ...           )                                           |                                                          |
| >>> c = ProcessingMechanism(                              |                                                          |
| ...           name='C',                                   |                                                          |
| ...           input_shapes=2,                                     |                                                          |
| ...           function=Logistic(gain=pnl.CONTROL)         |                                                          |
| ...           )                                           |                                                          |
| >>> comp = Composition(                                   |                                                          |
| ...           name='Comp',                                |                                                          |
| ...           enable_controller=True                      |                                                          |
| ...           )                                           |                                                          |
| >>> comp.add_linear_processing_pathway([a,c])             |                                                          |
| >>> comp.add_linear_processing_pathway([b,c])             |                                                          |
| >>> ctlr = OptimizationControlMechanism(                  |                                                          |
| ...            name='Controller',                         |                                                          |
| ...            monitor_for_control=[(pnl.MEAN, a)],       |                                                          |
| ...            control_signals=(GAIN, c),                 |                                                          |
| ...            agent_rep=comp                             |                                                          |
| ...            )                                          |                                                          |
| >>> comp.add_controller(ctlr)                             |                                                          |
+-----------------------------------------------------------+----------------------------------------------------------+

Note that the Composition's `controller <Composition.controller>` is not shown by default.  However this
can be shown, along with other information, using options in the Composition's `show_graph <ShowGraph.show_graph>`
method.  The figure below shows several examples.

.. _Composition_show_graph_options_figure:

**Output of show_graph using different options**

.. figure:: _static/Composition_show_graph_options_fig.svg
   :alt: Composition graph examples

   Displays of the Composition in the `example above <Composition_show_graph_basic_figure>`, generated using various
   options of its `show_graph <ShowGraph.show_graph>` method. **Panel A** shows the graph with its Projections labeled
   and Component dimensions displayed.  **Panel B** shows the `controller <Composition.controller>` for the
   Composition and its associated `ObjectiveMechanism` using the **show_controller** option (controller-related
   Components are displayed in blue by default).  **Panel C** adds the Composition's `CompositionInterfaceMechanisms
   <CompositionInterfaceMechanism>` using the **show_cim** option. **Panel D** shows a detailed view of the Mechanisms
   using the **show_node_structure** option, that includes their `Ports <Port>` and their `roles <NodeRole>` in the
   Composition. **Panel E** shows an even more detailed view using **show_node_structure** as well as **show_cim**.

If a Composition has one ore more Compositions nested as Nodes within it, these can be shown using the
**show_nested** option. For example, the pathway in the script below contains a sequence of Mechanisms
and nested Compositions in an outer Composition, ``comp``:

.. _Composition_show_graph_show_nested_figure:

+------------------------------------------------------+---------------------------------------------------------------+
| >>> mech_stim = ProcessingMechanism(name='STIMULUS') |.. figure:: _static/Composition_show_graph_show_nested_fig.svg |
| >>> mech_A1 = ProcessingMechanism(name='A1')         |                                                               |
| >>> mech_B1 = ProcessingMechanism(name='B1')         |                                                               |
| >>> comp1 = Composition(name='comp1')                |                                                               |
| >>> comp1.add_linear_processing_pathway([mech_A1,    |                                                               |
| ...                                      mech_B1])   |                                                               |
| >>> mech_A2 = ProcessingMechanism(name='A2')         |                                                               |
| >>> mech_B2 = ProcessingMechanism(name='B2')         |                                                               |
| >>> comp2 = Composition(name='comp2')                |                                                               |
| >>> comp2.add_linear_processing_pathway([mech_A2,    |                                                               |
| ...                                      mech_B2])   |                                                               |
| >>> mech_resp = ProcessingMechanism(name='RESPONSE') |                                                               |
| >>> comp = Composition()                             |                                                               |
| >>> comp.add_linear_processing_pathway([mech_stim,   |                                                               |
| ...                                     comp1, comp2,|                                                               |
| ...                                     mech_resp])  |                                                               |
| >>> comp.show_graph(show_nested=True)                |                                                               |
+------------------------------------------------------+---------------------------------------------------------------+


.. _ShowGraph_Class_Reference:

Class Reference
---------------

"""

import inspect
import pathlib
import site
import warnings
from psyneulink._typing import Union

import numpy as np
from beartype import beartype

import psyneulink
from psyneulink._typing import Optional, Union, Literal
from PIL import Image

from psyneulink.core.components.component import Component
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import \
    AGENT_REP, RANDOMIZATION_CONTROL_SIGNAL
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.port import Port
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ALL, BOLD, COMPONENT, COMPOSITION, CONDITIONS, FUNCTIONS, INSET, LABELS, LEARNABLE, \
    MECHANISM, MECHANISMS, NESTED, PROJECTION, PROJECTIONS, ROLES, SIMULATIONS, VALUES
from psyneulink.core.globals.utilities import convert_to_list

__all__ = ['DURATION', 'EXECUTION_SET', 'INITIAL_FRAME', 'MOVIE_DIR', 'MOVIE_NAME', 'MECH_FUNCTION_PARAMS',
           'NUM_TRIALS', 'NUM_RUNS', 'PORT_FUNCTION_PARAMS', 'SAVE_IMAGES',
           'SHOW', 'SHOW_CIM', 'SHOW_CONTROLLER', 'SHOW_JUST_LEARNING_PROJECTIONS', 'SHOW_LEARNING',
           'SHOW_PROJECTIONS_NOT_IN_COMPOSITION',
           'ShowGraph', 'UNIT',]


# Arguments passed to each nested Composition
SHOW_NODE_STRUCTURE = 'show_node_structure'
NODE_STRUCT_ARGS = 'node_struct_args'
# Options for show_node_structure argument of show_graph()
MECH_FUNCTION_PARAMS = "MECHANISM_FUNCTION_PARAMS"
PORT_FUNCTION_PARAMS = "PORT_FUNCTION_PARAMS"

SHOW_NESTED = 'show_nested'
SHOW_NESTED_ARGS = 'show_nested_args'
SHOW_CIM = 'show_cim'
SHOW_CONTROLLER = 'show_controller'
SHOW_LEARNING = 'show_learning'
SHOW_HEADERS = 'show_headers'
SHOW_TYPES = 'show_types'
SHOW_DIMENSIONS = 'show_dimensions'
SHOW_PROJECTION_LABELS = 'show_projection_labels'
SHOW_PROJECTIONS_NOT_IN_COMPOSITION = 'show_projections_not_in_composition'
SHOW_JUST_LEARNING_PROJECTIONS = 'show_just_learning_projections'
ACTIVE_ITEMS = 'active_items'
OUTPUT_FMT = 'output_fmt'

# show_graph animation options
NUM_TRIALS = 'num_trials'
NUM_RUNS = 'num_Runs'
UNIT = 'unit'
DURATION = 'duration'
MOVIE_DIR = 'movie_dir'
MOVIE_NAME = 'movie_name'
SAVE_IMAGES = 'save_images'
SHOW = 'show'
INITIAL_FRAME = 'INITIAL_FRAME'
EXECUTION_SET = 'EXECUTION_SET'

# Values for nested Compositions (passed from level to level)
ENCLOSING_COMP = 'enclosing_comp' # enclosing composition
NESTING_LEVEL = 'nesting_level'
NUM_NESTING_LEVELS = 'num_nesting_levels'
COMP_HIERARCHY = 'comp_hierarchy' # dict specifying the enclosing composition at each level of nesting


default_showgraph_subdir = 'pnl-show_graph-output'


def get_default_showgraph_dir():
    pnl_module_dir = pathlib.Path(psyneulink.__file__).parent.absolute()
    try:
        site_packages_dirs = site.getsitepackages()
    except AttributeError:
        # virtualenv <20 overrides site and has no getsitepackages
        site_packages_dirs = []

    # if psyneulink is installed in site-packages (not local/editable),
    # don't put show_graph files there
    for d in site_packages_dirs:
        if pathlib.Path(d) in pnl_module_dir.parents:
            default_dir = pathlib.Path('.')
            break
    else:
        default_dir = pnl_module_dir.parent

    default_dir = default_dir.joinpath(default_showgraph_subdir)

    return default_dir


class ShowGraphError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value


class ShowGraph():
    """ShowGraph object with `show_graph <ShowGraph.show_graph>` method for displaying `Composition`.

    Every Composition is assigned a ShowGraph object, with its `show_graph <ShowGraph.show_graph>` method
    assigned to, and callable as Composition.show_graph().

    Arguments
    ---------

    COMMENT:
    NOT FOR USER'S EYES!
    composition : Composition
        specifies the `Composition` to which the instance of ShowGraph is assigned.
    COMMENT

    direction : keyword : default 'BT'
        specifies the orientation of the graph (input -> output):
        - 'BT': bottom to top;
        - 'TB': top to bottom;
        - 'LR': left to right;
        - 'RL`: right to left.

    mechanism_shape : keyword : default 'oval'
        specifies the display shape of nodes that are not assigned a `NodeRole` associated with a dedicated shape.

    feedback_shape : keyword : default 'septagon'
        specifies the display shape of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.

    cycle_shape : keyword : default 'doublecircle'
        specifies the display shape of nodes that are assigned the `NodeRole` `CYCLE`.

    cim_shape : default 'square'
        specifies the shape in which `CompositionInterfaceMechanism`\\s are displayed.

    controller_shape : default 'doubleoctagon'
        specifies the shape in which a Composition's `controller <Composition.controller>` is displayed.

    composition_shape : default 'rectangle'
        specifies the shape in which nodes that represent `nested Compositions <Composition_Nested>` are displayed
        when **show_nested** is specified as False or a `Composition is nested <Composition_Nested>` below the
        level specified in a call to `show_graph() <ShowGraph.show_graph>`.

    agent_rep_shape : default 'egg'
        specifies the shape in which the `agent_rep` of an `OptimizationControlMechanism` is displayed.

    default_projection_arrow : keywrod : default 'normal'
         specifies the shape of the arrow used to display `MappingProjection`\\s.

    learning_projection_shape : default 'diamond'
        specifies the shape in which `LearningProjetions`\\s are displayed.

    control_projection_arrow : default 'box'
        specifies the shape in which the head of a `ControlProjection` is displayed.

    default_node_color : keyword : default 'black'
        specifies the color in which nodes not assigned another color are displayed.

    active_color : keyword : default BOLD
        specifies how to highlight the item(s) specified in the **active_items** argument of a call to `show_graph
        <ShowGraph.show_graph>`:  either a color recognized by GraphViz, or the keyword *BOLD*.

    input_color : keyword : default 'green',
        specifies the color in which `INPUT <NodeRole.INPUT>` Nodes of the Composition are displayed.

    probe_color : keyword : default 'pink',
        specifies the color in which `PROBE <NodeRole.PROBE>` Nodes of the Composition are displayed.

    output_color : keyword : default 'red',
        specifies the color in which `OUTPUT <NodeRole.OUTPUT>` Nodes of the Composition are displayed.

    input_and_output_color : keyword : default 'brown'
        specifies the color in which nodes that are both an `INPUT <NodeRole.INPUT>` and an `OUTPUT
        <NodeRole.OUTPUT>` Node of the Composition are displayed.

    COMMENT:
    feedback_color : keyword : default 'yellow'
        specifies the display color of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.
    COMMENT

    control_color : keyword : default 'blue'
        specifies the color in which `ControlMechanisms <ControlMechanism>` (other than a Composition's
        `controller <Composition.controller>` and `ControlProjections <ControlProjection>` are displayed.

    controller_color : keyword : default 'purple'
        specifies the color in which a Composition's `controller <Composition.controller>` is displayed.

    learning_color : keyword : default 'orange'
        specifies the color in which the `learning components <Composition_Learning_Components>` are displayed.

    composition_color : keyword : default 'pink'
        specifies the color in which nodes that represent `nested Compositions <Composition_Nested> are displayed
        when **show_nested** is specified as False or a `Composition is nested <Composition_Nested>` below the
        level specified in a call to `show_graph() <ShowGraph.show_graph>`.

    inactive_projection_color : keyword : default 'red'
        specifies the color in which `Projections <Projection>` not active within the `Composition` are displayed,
        when the `show_projections_not_in_composition <ShowGraph.show_projections_not_in_composition>` option is True.

    default_width : int : default 1
        specifies the width to use for the outline of nodes and the body of Projection arrows.

    active_thicker_by : int : default 2
        specifies the amount by which to increase the width of the outline of Components specified in the
        **active_items** argument of a call to `show_graph() <ShowGraph.show_graph>`.

    bold_width : int : default 3,
        specifies the width of the outline for `INPUT` and `OUTPUT` Nodes of the Composition.

    COMMENT:
    input_rank : keyword : default 'source',

    control_rank : keyword : default 'min',

    learning_rank : keyword : default 'min',

    output_rank : keyword : default 'max'
    COMMENT

    """

    def __init__(self,
                 composition,
                 direction: Literal['BT', 'TB', 'LR', 'RL'] = 'BT',
                 # Node shapes:
                 mechanism_shape = 'oval',
                 feedback_shape = 'octagon',
                 cycle_shape = 'doublecircle',
                 cim_shape = 'rectangle',
                 controller_shape = 'doubleoctagon',
                 composition_shape = 'rectangle',
                 agent_rep_shape = 'egg',
                 # Projection shapes
                 default_projection_arrow = 'normal',
                 learning_projection_shape = 'diamond',
                 control_projection_arrow='box',
                 # Colors:
                 default_node_color = 'black',
                 active_color=BOLD,
                 input_color='green',
                 probe_color='pink',
                 output_color='red',
                 input_and_output_color='brown',
                 # feedback_color='yellow',
                 control_color='blue',
                 controller_color='purple',
                 learning_color='orange',
                 composition_color='pink',
                 inactive_projection_color='red',
                 # Lines:
                 default_width = 1,
                 active_thicker_by = 2,
                 bold_width = 3,
                 # Order:
                 input_rank = 'source',
                 control_rank = 'min',
                 learning_rank = 'min',
                 output_rank = 'max',
                ):

        self.composition = composition
        self.direction = direction

        # Node shapes:
        self.mechanism_shape = mechanism_shape
        self.feedback_shape = feedback_shape
        self.cycle_shape = cycle_shape
        self.struct_shape = 'plaintext' # assumes use of html
        self.cim_shape = cim_shape
        self.composition_shape = composition_shape
        self.controller_shape = controller_shape
        self.agent_rep_shape = agent_rep_shape
        # Projection shapes
        self.learning_projection_shape = learning_projection_shape
        self.control_projection_arrow =control_projection_arrow
        # Colors:
        self.default_node_color = default_node_color
        self.active_color = active_color
        self.input_color = input_color
        self.probe_color=probe_color
        self.output_color = output_color
        self.input_and_output_color = input_and_output_color
        # self.feedback_color = self.feedback_color
        self.control_color =control_color
        self.controller_color =controller_color
        self.learning_color =learning_color
        self.composition_color =composition_color
        self.inactive_projection_color =inactive_projection_color
        # Lines:
        self.default_projection_arrow = default_projection_arrow
        self.default_width = default_width
        self.active_thicker_by = active_thicker_by
        self.bold_width = bold_width
        # Order:
        self.input_rank = input_rank
        self.control_rank = control_rank
        self.learning_rank = learning_rank
        self.output_rank = output_rank

    @beartype
    @handle_external_context(source=ContextFlags.COMPOSITION)
    def show_graph(self,
                   show_all: bool = False,
                   show_node_structure: Union[bool, Literal['values', 'labels', 'functions', 'MECHANISM_FUNCTION_PARAMS',
                                                             'PORT_FUNCTION_PARAMS', 'roles', 'all']] = False,
                   show_nested: Optional[Union[bool, int, dict, Literal['nested', 'inset']]] = 'nested',
                   show_nested_args: Optional[Union[bool, dict, Literal['all']]] = 'all',
                   show_cim: bool = False,
                   show_controller: Union[bool, Literal['agent_rep']] = True,
                   show_learning: Union[bool, Literal[ALL, LEARNABLE, SHOW_JUST_LEARNING_PROJECTIONS]] = False,
                   show_headers: bool = True,
                   show_types: bool = False,
                   show_dimensions: bool = False,
                   show_projection_labels: bool = False,
                   show_projections_not_in_composition: bool = False,
                   active_items=None,
                   output_fmt: Optional[Literal['pdf', 'gv', 'jupyter', 'gif', 'source']] = 'pdf',
                   context=None,
                   *args,
                   **kwargs):
        """
        show_graph(                                  \
           show_all=False,                           \
           show_node_structure=False,                \
           show_nested=NESTED,                       \
           show_nested_args=ALL,                     \
           show_cim=False,                           \
           show_controller=True,                     \
           show_learning=False,                      \
           show_headers=True,                        \
           show_types=False,                         \
           show_dimensions=False,                    \
           show_projection_labels=False,             \
           show_projections_not_in_composition=False \
           active_items=None,                        \
           output_fmt='pdf',                         \
           context=None)

        Show graphical display of Components in a Composition's graph.  See `show_graph <ShowGraph_show_graph_Method>`
        for additional details.

        .. note::
           This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
           (standard with PsyNeuLink pip install)

        Arguments
        ---------

        show_all : bool : default False
            if False, defer to specification of all other arguments;  if True, override all show_XXX arguments,
            automatically specifying them with their most informative settings.

        show_node_structure : bool, VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS, PORT_FUNCTION_PARAMS, ROLES, \
        or ALL : default False
            show a detailed representation of each `Mechanism <Mechanism>` in the graph, including its `Ports <Port>`;
            can have any of the following settings alone or in a list:

            * `True` -- show Ports of Mechanism, but not information about the `value
              <Component.value>` or `function <Component.function>` of the Mechanism or its Ports.

            * *VALUES* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <Port_Base.value>` of each of its Ports.

            * *LABELS* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <Port_Base.value>` of each of its Ports, using any labels for the values of InputPorts and
              OutputPorts specified in the Mechanism's `input_labels_dict <Mechanism.input_labels_dict>` and
              `output_labels_dict <Mechanism.output_labels_dict>`, respectively.

            * *FUNCTIONS* -- show the `function <Mechanism_Base.function>` of the Mechanism and the `function
              <Port_Base.function>` of its InputPorts and OutputPorts.

            * *MECH_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
              Mechanism in the Composition (only applies if *FUNCTIONS* is True).

            * *PORT_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
              Port of each Mechanism in the Composition (only applies if *FUNCTIONS* is True).

            * *ROLES* -- show the `role <NodeRole>` of the Mechanism in the Composition
              (but not any of the other information;  use *ALL* to show ROLES with other information).

            * *ALL* -- shows the role, `function <Component.function>`, and `value <Component.value>` of the
              Mechanisms in the `Composition` and their `Ports <Port>` (using labels for
              the values, if specified -- see above), including parameters for all functions.

        show_nested : bool | int | NESTED | INSET : default NESTED
            specifies whether or not to show `nested Compositions <Composition_Nested>` and, if so, how many
            levels of nesting to show (*NESTED*, True or int) -- with Projections shown directly from Components
            in an enclosing Composition to and from ones in the nested Composition; or each nested Composition as
            a separate inset (*INSET*).  *NESTED* specifies all levels of nesting shown; 0 specifies none (same as
            False), and a non-zero integer species that number of nested levels to shown.  Compsitions nested at the
            specified level are shown as a node (pink box by default). and ones below the specified level are not
            shown at all.

        show_nested_args : bool | dict : default ALL
            specifies arguments in call to show_graph passed to `nested Composition(s) <Composition_Nested>` if
            **show_nested** is specified.  A dict can be used to specify any of the arguments allowed for
            show_graph to be used for the nested Composition(s);  *ALL* passes all arguments specified for the main
            Composition to the nested one(s);  True uses the default values of show_graph args for the nested
            Composition(s).

        show_cim : bool : default False
            specifies whether or not to show the Composition's `input_CIM <Composition.input_CIM>`, `parameter_CIM
            <Composition.parameter_CIM>`, and `output_CIM <Composition.output_CIM>` `CompositionInterfaceMechanisms
            <CompositionInterfaceMechanism>` (CIMs).

        show_controller :  bool or AGENT_REP : default True
            specifies whether or not to show the Composition's `controller <Composition.controller>` and associated
            `objective_mechanism <ControlMechanism.objective_mechanism>` if it has one.  If the controller is an
            OptimizationControlMechanism and it has an `agent_rep <OptimizationControlMechanism>`, then specifying
            *AGENT_REP* also shows that.  All control-related items are displayed in the color specified for
            **controller_color**.

        show_learning : bool, LEARNABLE, SHOW_JUST_LEARNING_PROJECTIONS, or ALL : default False
            specifies whether or not to show the `learning components <Composition_Learning_Components>` of the
            `Composition`; they are all displayed in the color specified for **learning_color**.
            Projections that receive a `LearningProjection` are shown as a diamond-shaped node.
            If set to *LEARNABLE*, all `LearningMechanism`\\s are shown but only `LearningProjection`\\s that are
            `enabled <LearningProjection.learning_enabled>` are shown; if set to True or *ALL*, all Projections
            associated with learning are shown (that is, the LearningProjections as well as those from
            `ProcessingMechanisms <ProcessingMechanism>` to `LearningMechanisms <LearningMechanism>`
            that convey error and activation information;  if set to *SHOW_JUST_LEARNING_PROJECTIONS*,
            the display is restricted to LearningProjections that are `enabled <LearningProjection.learning_enabled>`.

        show_projection_labels : bool : default False
            specifies whether or not to show names of projections.

        show_projections_not_in_composition : bool : default False
            specifies whether or not to show `Projections <Projection>` that are not active in the current
            `Composition` (and, accordingly, are *not* listed in its `projections <Composition.projections>`
            attribute);  these are shown in red.

        show_headers : bool : default True
            specifies whether or not to show headers in the subfields of a Mechanism's node;  only takes effect if
            **show_node_structure** is specified (see above).

        show_types : bool : default False
            specifies whether or not to show type (class) of `Mechanism <Mechanism>` in each node label.

        show_dimensions : bool : default False
            specifies whether or not to show dimensions for the `variable <Component.variable>` and `value
            <Component.value>` of each Component in the graph (and/or MappingProjections when show_learning
            is `True`);  can have the following settings:

            * *MECHANISMS* -- shows `Mechanism <Mechanism>` input and output dimensions.  Input dimensions are shown
              in parentheses below the name of the Mechanism; each number represents the dimension of the `variable
              <InputPort.variable>` for each `InputPort` of the Mechanism; Output dimensions are shown above
              the name of the Mechanism; each number represents the dimension for `value <OutputPort.value>` of each
              of `OutputPort` of the Mechanism.

            * *PROJECTIONS* -- shows `MappingProjection` `matrix <MappingProjection.matrix>` dimensions.  Each is
              shown in (<dim>x<dim>...) format;  for standard 2x2 "weight" matrix, the first entry is the number of
              rows (input dimension) and the second the number of columns (output dimension).

            * *ALL* -- eqivalent to `True`; shows dimensions for both Mechanisms and Projections (see above for
              formats).

        active_items : List[Component] : default None
            specifies one or more items in the graph to display in the color specified by *active_color**.

        output_fmt : keyword or None : default 'pdf'
            'pdf': generate and open a pdf with the visualization;
            'jupyter': return the object (for working in jupyter/ipython notebooks);
            'gv': return graphviz object
            'gif': return gif used for animation
            'source': return the source code for the graphviz object
            None : return None

        Returns
        -------

        `pdf` or Graphviz graph object :
            determined by **output_fmt**:
            - ``pdf`` -- PDF: (placed in current directory);
            - ``gv`` or ``jupyter`` -- Graphviz graph object;
            - ``gif`` -- gif
            - ``source`` -- str with content of G.body

        """
        from psyneulink.core.compositions.composition import Composition

        composition = self.composition

        if context.execution_id is None:
            context.execution_id = composition.default_execution_id

        # Args not specified by user but used in calls to show_graph for nested Compositions
        comp_hierarchy = kwargs.pop(COMP_HIERARCHY, {})
        enclosing_comp = kwargs.pop(ENCLOSING_COMP,None)
        nesting_level = kwargs.pop(NESTING_LEVEL,None)
        self.num_nesting_levels = kwargs.pop(NUM_NESTING_LEVELS,None)

        enclosing_g = enclosing_comp._show_graph.G if enclosing_comp else None
        processing_graph = self._get_processing_graph(composition, context)
        # IMPLEMENTATION_NOTE:  Take diff with following to get scheduling edges not in compostion graph:
        # processing_graph = composition.scheduler.dependency_dict

        # Validate active_items  ~~~~~~~~~~~~~~~~~~~~~~~~~
        active_items = active_items or []
        if active_items:
            active_items = convert_to_list(active_items)
            if (composition.scheduler.get_clock(context).time.run >= composition._animate_num_runs or
                    composition.scheduler.get_clock(context).time.trial >= composition._animate_num_trials):
                return
            for item in active_items:
                if not isinstance(item, Component) and item is not INITIAL_FRAME:
                    raise ShowGraphError(
                        f"PROGRAM ERROR: Item ({item}) specified in 'active_items' argument for 'show_graph' method of "
                        f"{composition.name} is not a { Component.__name__}.")
        composition.active_item_rendered = False

        # ASSIGN ATTRIBUTES PASSED TO NESTED COMPOSITIONS  -----------------------------------------------

        if show_all:
            show_node_structure=ALL
            show_nested=NESTED
            show_nested_args=ALL
            show_cim=True
            show_controller=True
            show_learning=ALL
            show_headers=True
            show_projections_not_in_composition=True

        # Assign node_struct_arg based on show_node_structure ~~~~~~~~~~~~~~~~~~~~~~~~~
        # Argument values used to call Mechanism._show_structure()
        if isinstance(show_node_structure, (list, tuple, set)):
            node_struct_args = {'composition': self.composition,
                                'show_roles': any(key in show_node_structure for key in {ROLES, ALL}),
                                'show_conditions': any(key in show_node_structure for key in {CONDITIONS, ALL}),
                                'show_functions': any(key in show_node_structure for key in {FUNCTIONS, ALL}),
                                'show_mech_function_params': any(key in show_node_structure
                                                                 for key in {MECH_FUNCTION_PARAMS, ALL}),
                                'show_port_function_params': any(key in show_node_structure
                                                                  for key in {PORT_FUNCTION_PARAMS, ALL}),
                                'show_values': any(key in show_node_structure for key in {VALUES, ALL}),
                                'use_labels': any(key in show_node_structure for key in {LABELS, ALL}),
                                'show_headers': show_headers,
                                'output_fmt': 'struct',
                                'context':context}
        else:
            node_struct_args = {'composition': self.composition,
                                'show_roles': show_node_structure in {ROLES, ALL},
                                'show_conditions': show_node_structure in {CONDITIONS, ALL},
                                'show_functions': show_node_structure in {FUNCTIONS, ALL},
                                'show_mech_function_params': show_node_structure in {MECH_FUNCTION_PARAMS, ALL},
                                'show_port_function_params': show_node_structure in {PORT_FUNCTION_PARAMS, ALL},
                                'show_values': show_node_structure in {VALUES, LABELS, ALL},
                                'use_labels': show_node_structure in {LABELS, ALL},
                                'show_headers': show_headers,
                                'output_fmt': 'struct',
                                'context': context}

        # Assign num_nesting_levels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  (note:  this is assigned to self since it applies to all levels)
        # For outermost Composition:
        # - initialize nesting level
        # - set num_nesting_levels
        if enclosing_comp is None:
            # initialize nesing_level
            nesting_level = 0
            # show_nested specified number of nested levels to show, so set to that
            if type(show_nested) is int:
                self.num_nesting_levels = show_nested
            # only allow outermost Composition (first nested layer showed as Composition node, none others shown
            elif show_nested is False:
                self.num_nesting_levels = 0
            # allow arbitrary number of nesting_levels
            elif show_nested is NESTED:
                self.num_nesting_levels = float("inf")
            else:
                self.num_nesting_levels = None

        # Assign show_nested  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If num_nesting_levels is specified, set show_nested based on current nesting_level
        if self.num_nesting_levels is not None:
            if nesting_level < self.num_nesting_levels:
                show_nested = NESTED
            else:
                show_nested = False
        # Otherwise, set show_nested as NESTED unless it was specified as INSET
        elif show_nested and show_nested != INSET:
            show_nested = NESTED

        # Assign nested_args  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # to be passed in call to show_graph for nested Composition(s)
        # Get args passed in from main call to show_graph (to be passed to helper methods)
        show_graph_args = locals().copy()
        # Update any modified above
        nested_args = show_nested_args or {}
        if nested_args == ALL:
            # Use show_graph args (passed in from main call to show_graph, updated as above)
            nested_args = dict({k:show_graph_args[k] for k in list(inspect.signature(self.show_graph).parameters)})
        nested_args[ACTIVE_ITEMS] = active_items
        nested_args[NODE_STRUCT_ARGS] = node_struct_args
        nested_args[SHOW_NESTED] = show_nested
        nested_args[NUM_NESTING_LEVELS] = self.num_nesting_levels

        # BUILD GRAPH ------------------------------------------------------------------------

        import graphviz as gv
        self.style = 'solid'

        G = gv.Digraph(
            name=composition.name,
            engine="dot",
            node_attr={
                'fontsize': '12',
                'fontname': 'arial',
                'shape': 'record',
                'color': self.default_node_color,
                'penwidth': str(self.default_width),
            },
            edge_attr={
                'fontsize': '10',
                'fontname': 'arial'
            },
            graph_attr={
                "rankdir": self.direction,
                'overlap': "False",
                'label' : composition.name,
                # 'newrank': "True"
            },
        )
        self.G = G

        # get all Nodes
        if output_fmt != 'gv':
            composition._analyze_graph(context=context)
            if composition._need_check_for_unused_projections:
                composition._check_for_unused_projections(context)

        rcvrs = list(processing_graph.keys())
        for rcvr in rcvrs:

            # If show_controller is true, objective mechanism is handled in _assign_controller_components
            if (show_controller
                and composition.controller
                and composition.controller.objective_mechanism
                and rcvr is composition.controller.objective_mechanism):
                continue

            self._assign_processing_components(G,
                                               rcvr,
                                               processing_graph,
                                               enclosing_comp,
                                               comp_hierarchy,
                                               nesting_level,
                                               active_items,
                                               show_nested,
                                               show_cim,
                                               show_learning,
                                               show_node_structure,
                                               node_struct_args,
                                               show_types,
                                               show_dimensions,
                                               show_projection_labels,
                                               show_projections_not_in_composition,
                                               nested_args,
                                               context)

        # Add cim Components to graph if show_cim
        if show_cim:
            self._assign_cim_components(G,
                                        enclosing_comp,
                                        active_items,
                                        show_nested,
                                        show_types,
                                        show_dimensions,
                                        show_node_structure,
                                        node_struct_args,
                                        show_projection_labels,
                                        show_projections_not_in_composition,
                                        show_controller,
                                        comp_hierarchy,
                                        context)

        # Add controller-related Components to graph if show_controller
        if show_controller:
            self._assign_controller_components(G,
                                               active_items,
                                               show_nested,
                                               show_cim,
                                               show_controller,
                                               show_learning,
                                               show_types,
                                               show_dimensions,
                                               show_node_structure,
                                               node_struct_args,
                                               show_projection_labels,
                                               show_projections_not_in_composition,
                                               comp_hierarchy,
                                               nesting_level,
                                               context)

        # Add learning-related Components to graph if show_learning
        if show_learning:
            self._assign_learning_components(G,
                                             processing_graph,
                                             enclosing_comp,
                                             comp_hierarchy,
                                             nesting_level,
                                             active_items,
                                             show_nested,
                                             show_cim,
                                             show_learning,
                                             show_types,
                                             show_dimensions,
                                             show_node_structure,
                                             node_struct_args,
                                             show_projection_labels,
                                             show_projections_not_in_composition,
                                             context)

        return self._generate_output(G,
                                     enclosing_comp,
                                     active_items,
                                     show_controller,
                                     output_fmt,
                                     context)

    def __call__(self, **args):
        return self.show_graph(**args)

    def _get_processing_graph(self, composition, context):
        """Helper method that allows override by subclass to filter nodes and their dependencies used for graph"""
        return composition.graph_processing.dependency_dict

    def _get_nodes(self, composition ,context):
        """Helper method that allows override by subclass to filter nodes used for graph"""
        return composition.nodes

    def _get_projections(self, composition, context):
        """Helper method that allows override by subclass to filter projections used for graph"""
        return composition.projections

    def _proj_in_composition(self, proj, composition_projections, context):
        """Helper method that allows override by subclass to filter projections used for graph"""
        return proj in composition_projections

    def _get_roles_by_node(self, composition, node, context):
        """Helper method that allows override by subclass to filter NodeRoles used for graph"""
        return composition.get_roles_by_node(node)

    def _get_nodes_by_role(self, composition, role, context):
        """Helper method that allows override by subclass to filter NodeRoles used for graph"""
        return composition.get_nodes_by_role(role)

    def _implement_graph_node(self, graph, rcvr, context, *args, **kwargs):
        """Helper method that allows override by subclass to assign custom attributes to nodes"""
        graph.node(*args, **kwargs)

    def _implement_graph_edge(self, graph, proj, context, *args, **kwargs):
        """Helper method that allows override by subclass to assign custom attributes to edges"""
        graph.edge(*args, **kwargs)

    def _assign_processing_components(self,
                                      g,
                                      rcvr,
                                      processing_graph,
                                      enclosing_comp,
                                      comp_hierarchy,
                                      nesting_level,
                                      active_items,
                                      show_nested,
                                      show_cim,
                                      show_learning,
                                      show_node_structure,
                                      node_struct_args,
                                      show_types,
                                      show_dimensions,
                                      show_projection_labels,
                                      show_projections_not_in_composition,
                                      nested_args,
                                      context):
        """Assign nodes to graph"""

        from psyneulink.core.compositions.composition import Composition, NodeRole

        composition = self.composition

        # User passed attrs for nested Composition
        if isinstance(rcvr, Composition):
            if show_nested:
                comp_hierarchy.update({nesting_level:composition})
                nested_args.update({OUTPUT_FMT:'gv',
                                    COMP_HIERARCHY:comp_hierarchy,
                                    # 'composition': rcvr,
                                    ENCLOSING_COMP:composition,
                                    NESTING_LEVEL:nesting_level + 1,
                                    })
                # Get subgraph for nested Composition
                # IMPLEMENTATION NOTE: FIX: HACK SO NESTED COMPOSITIONS DON'T CRASH ANIMATION (THOUGH STILL NOT SHOWN)
                if hasattr(composition, '_animate') and composition._animate is not False:
                    rcvr._animate = composition._animate
                    rcvr._set_up_animation(context)
                    rcvr._animate_num_trials = composition._animate_num_trials + 1
                nested_comp_graph = rcvr._show_graph.show_graph(**nested_args)

                nested_comp_graph.name = "cluster_" + rcvr.name
                rcvr_label = rcvr.name

                # Assign color to nested_comp, including highlighting if it is the active_item
                # if rcvr in composition.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
                #     nested_comp_graph.attr(color=feedback_color)
                # nested_comp_attributes = {"label":rcvr_label}
                nested_comp_attributes = {}
                input_nodes = self._get_nodes_by_role(composition, NodeRole.INPUT, context)
                output_nodes = self._get_nodes_by_role(composition, NodeRole.OUTPUT, context)
                probe_nodes = self._get_nodes_by_role(composition, NodeRole.PROBE, context)
                if rcvr in input_nodes and output_nodes:
                    nested_comp_attributes.update({"color": self.input_and_output_color})
                elif rcvr in input_nodes:
                    nested_comp_attributes.update({"color": self.input_color})
                elif rcvr in probe_nodes:
                    nested_comp_attributes.update({"color": self.probe_color})
                elif rcvr in output_nodes:
                    nested_comp_attributes.update({"color": self.output_color})
                if rcvr in active_items:
                    if self.active_color != BOLD:
                        nested_comp_attributes.update({"color": self.active_color})
                    nested_comp_attributes.update({"penwidth": str(self.default_width + self.active_thicker_by)})
                    composition.active_item_rendered = True
                nested_comp_graph.attr(**nested_comp_attributes)
                nested_comp_graph.attr(label=rcvr_label)
                g.subgraph(nested_comp_graph)

                if show_nested is NESTED:
                    return

        # DEAL WITH LEARNING
        # If rcvr is a learning component and not an INPUT node,
        #    break and handle in _assign_learning_components()
        #    (node: this allows TARGET node for learning to remain marked as an INPUT node)
        if (NodeRole.LEARNING in self._get_roles_by_node(composition, rcvr, context)):
            return

        # DEAL WITH CONTROLLER's OBJECTIVEMECHANIMS
        # If rcvr is ObjectiveMechanism for Composition's controller,
        #    break and handle in _assign_controller_components()
        if (isinstance(rcvr, ObjectiveMechanism)
                and composition.controller
                and rcvr is composition.controller.objective_mechanism):
            return

        # IMPLEMENT RECEIVER NODE:
        #    set rcvr shape, color, and penwidth based on node type
        rcvr_rank = 'same'

        # SET SPECIAL SHAPES

        # Cycle or Feedback Node
        if isinstance(rcvr, Composition):
            node_shape = self.composition_shape
        elif rcvr in self._get_nodes_by_role(composition, NodeRole.FEEDBACK_SENDER, context):
            node_shape = self.feedback_shape
        elif rcvr in self._get_nodes_by_role(composition, NodeRole.CYCLE, context):
            node_shape = self.cycle_shape
        else:
            node_shape = self.mechanism_shape

        # SET STROKE AND COLOR
        #    Based on Input, Output, Composition and/or Active

        # Get condition if any associated with rcvr
        if rcvr in composition.scheduler.conditions:
            condition = composition.scheduler.conditions[rcvr]
        else:
            condition = None

        # INPUT and OUTPUT Node
        if rcvr in composition.get_nodes_by_role(NodeRole.INPUT) and \
                rcvr in composition.get_nodes_by_role(NodeRole.OUTPUT):
            if rcvr in active_items:
                if self.active_color == BOLD:
                    rcvr_color = self.input_and_output_color
                else:
                    rcvr_color = self.active_color
                rcvr_penwidth = str(self.bold_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = self.input_and_output_color
                rcvr_penwidth = str(self.bold_width)

        # INPUT Node
        elif rcvr in composition.get_nodes_by_role(NodeRole.INPUT):
            if rcvr in active_items:
                if self.active_color == BOLD:
                    rcvr_color = self.input_color
                else:
                    rcvr_color = self.active_color
                rcvr_penwidth = str(self.bold_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = self.input_color
                rcvr_penwidth = str(self.bold_width)
            rcvr_rank = self.input_rank

        # PROBE Node
        elif rcvr in composition.get_nodes_by_role(NodeRole.PROBE):
            if rcvr in active_items:
                if self.active_color == BOLD:
                    rcvr_color = self.probe_color
                else:
                    rcvr_color = self.active_color
                rcvr_penwidth = str(self.bold_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = self.probe_color
                rcvr_penwidth = str(self.bold_width)
            rcvr_rank = self.output_rank

        # OUTPUT Node
        elif rcvr in composition.get_nodes_by_role(NodeRole.OUTPUT):
            if rcvr in active_items:
                if self.active_color == BOLD:
                    rcvr_color = self.output_color
                else:
                    rcvr_color = self.active_color
                rcvr_penwidth = str(self.bold_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = self.output_color
                rcvr_penwidth = str(self.bold_width)
            rcvr_rank = self.output_rank

        # CONTROL Node
        elif isinstance(rcvr, ControlMechanism):
            if rcvr in active_items:
                if self.active_color == BOLD:
                    rcvr_color = self.control_color
                else:
                    rcvr_color = self.active_color
                rcvr_penwidth = str(self.bold_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = self.control_color
                rcvr_penwidth = str(self.bold_width)
            rcvr_rank = self.output_rank

        # Composition that is neither INPUT, OUTPUT or CONTROL Node
        elif isinstance(rcvr, Composition) and show_nested is not NESTED:
            if rcvr in active_items:
                if self.active_color == BOLD:
                    rcvr_color = self.composition_color
                else:
                    rcvr_color = self.active_color
                rcvr_penwidth = str(self.bold_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = self.composition_color
                rcvr_penwidth = str(self.bold_width)

        # Active Node that is none of the above
        elif rcvr in active_items:
            if self.active_color == BOLD:
                rcvr_color = self.default_node_color
            else:
                rcvr_color = self.active_color
            rcvr_penwidth = str(self.default_width + self.active_thicker_by)
            composition.active_item_rendered = True

        # Inactive Node that is none of the above
        else:
            rcvr_color = self.default_node_color
            rcvr_penwidth = str(self.default_width)

        # Implement rcvr node
        rcvr_label = self._get_graph_node_label(composition,
                                                rcvr,
                                                show_types,
                                                show_dimensions)

        if show_node_structure and isinstance(rcvr, Mechanism):
            args = (rcvr_label, rcvr._show_structure(**node_struct_args,
                                                     node_border=rcvr_penwidth,
                                                     condition=condition))
            kwargs = {'shape': self.struct_shape,
                          'color':rcvr_color,
                          'penwidth': rcvr_penwidth,
                          'rank': rcvr_rank}
        else:
            args = (rcvr_label,)
            kwargs = {'shape': node_shape,
                          'color':rcvr_color,
                          'penwidth': rcvr_penwidth,
                          'rank': rcvr_rank}

        self._implement_graph_node(g, rcvr, context,*args, **kwargs)

        # 7/9/24
        # FIX: IMPLEMENT THIS AS METHOD THAT CAN BE OVERRIDEN BY SUBCLASS TO IMPLEMENT DIRECT PROJS TO NESTED NODES
        # Implement sender edges from Nodes within Composition
        sndrs = processing_graph[rcvr]
        self._assign_incoming_edges(g,
                                    rcvr,
                                    rcvr_label,
                                    sndrs,
                                    active_items,
                                    show_nested,
                                    show_cim,
                                    show_learning,
                                    show_types,
                                    show_dimensions,
                                    show_node_structure,
                                    show_projection_labels,
                                    show_projections_not_in_composition,
                                    enclosing_comp=enclosing_comp,
                                    comp_hierarchy=comp_hierarchy,
                                    nesting_level=nesting_level,
                                    context=context)

    def _assign_cim_components(self,
                               g,
                               enclosing_comp,
                               active_items,
                               show_nested,
                               show_types,
                               show_dimensions,
                               show_node_structure,
                               node_struct_args,
                               show_projection_labels,
                               show_projections_not_in_composition,
                               show_controller,
                               comp_hierarchy,
                               context):

        from psyneulink.core.compositions.composition import Composition, NodeRole
        composition = self.composition
        composition_nodes = self._get_nodes(composition, context)
        composition_projections = self._get_projections(composition, context)
        enclosing_g = enclosing_comp._show_graph.G if enclosing_comp else None
        enclosing_comp_projections = self._get_projections(enclosing_comp, context) if enclosing_comp else None

        cim_rank = 'same'

        def _render_projection(_g, proj, sndr_label, rcvr_label,
                               proj_color=self.default_node_color,
                               arrowhead=self.default_projection_arrow,
                               style=self.style):
            if any(item in active_items for item in {proj, proj.sender.owner}):
                if self.active_color == BOLD:
                    color = proj_color
                else:
                    color = self.active_color
                proj_width = str(self.default_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                color = proj_color
                proj_width = str(self.default_width)

            if show_projection_labels:
                label = self._get_graph_node_label(composition, proj, show_types, show_dimensions)
            else:
                label = ''

            _g.edge(sndr_label, rcvr_label,
                    label=label, color=color,
                    penwidth=proj_width,
                    arrowhead=arrowhead,
                    style=style)

        for cim in composition.cims:

            # Skip cim if it is not doing anything
            if not (cim.afferents or cim.efferents):
                continue

            # ASSIGN CIM NODE ****************************************************************

            # Assign Node attributes

            # Also take opportunity to validate that cim is input_CIM, parameter_CIM or output_CIM
            if cim is composition.input_CIM:
                cim_type_color = self.input_color
            elif cim is composition.parameter_CIM:
                # Set default parameter_CIM color to control_color
                cim_type_color = self.control_color
                # But if any Projection to it is from a controller, use controller_color
                for input_port in cim.input_ports:
                    for proj in input_port.path_afferents:
                        if (proj not in enclosing_comp_projections and not show_projections_not_in_composition):
                            continue
                        if self._trace_senders_for_controller(proj, context, enclosing_comp):
                            cim_type_color = self.controller_color
            elif cim is composition.output_CIM:
                cim_type_color = self.output_color
            else:
                assert False, \
                    f'PROGRAM ERROR: _assign_cim_components called with node ' \
                    f'that is not input_CIM, parameter_CIM, or output_CIM'

            cim_penwidth = str(self.default_width)

            if cim in active_items:
                if self.active_color == BOLD:
                    cim_color = cim_type_color
                else:
                    cim_color = self.active_color
                cim_penwidth = str(self.default_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                cim_color = cim_type_color

            compact_cim = not cim.afferents or not enclosing_g or show_nested is INSET

            # Create CIM node
            cim_label = self._get_graph_node_label(composition, cim, show_types, show_dimensions)
            if show_node_structure:
                g.node(cim_label,
                       cim._show_structure(**node_struct_args,
                                           node_border=cim_penwidth,
                                           compact_cim=compact_cim),
                       shape=self.struct_shape,
                       color=cim_color,
                       rank=cim_rank,
                       penwidth=cim_penwidth)

            else:
                g.node(cim_label,
                       shape=self.cim_shape,
                       color=cim_color,
                       rank=cim_rank,
                       penwidth=cim_penwidth)

            # FIX 6/2/20:  THIS CAN BE CONDENSED (ABSTRACTED INTO GENERIC FUNCTION TAKING cim-SPECIFIC PARAMETERS)
            # ASSIGN CIM PROJECTIONS ****************************************************************

            # INPUT_CIM -----------------------------------------------------------------------------

            if cim is composition.input_CIM:

                # Projections from Node(s) in enclosing Composition to input_CIM
                for input_port in composition.input_CIM.input_ports:
                    projs = input_port.path_afferents
                    for proj in projs:

                        proj_color=self.default_node_color
                        if proj not in enclosing_comp_projections:
                            if not show_projections_not_in_composition:
                                continue
                            else:
                                proj_color=self.inactive_projection_color

                        # Get label for Node that sends the input (sndr_label)
                        sndr_node_output_port = proj.sender
                        # Skip if sender is a CIM (handled by enclosing Composition's call to this method)
                        if isinstance(sndr_node_output_port.owner, CompositionInterfaceMechanism):
                            continue
                        # Skip if there is no outer Composition (enclosing_g),
                        #    or Projections between Compositions are not being shown (show_nested=INSET)
                        if not enclosing_g or show_nested is INSET:
                            continue
                        sndr_node_output_port_owner = sndr_node_output_port.owner

                        sndr_label = self._get_graph_node_label(composition,
                                                           sndr_node_output_port_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for CIM's port as edge's receiver
                            rcvr_cim_proj_label = f"{cim_label}:{type(input_port).__name__}-{proj.receiver.name}"
                            if (isinstance(sndr_node_output_port_owner, Composition)
                                    and show_nested is not NESTED):
                                sndr_output_node_proj_label = sndr_label
                            else:
                                # Need to use direct reference to proj.sender rather than snder_input_node
                                #    since could be Composition, which does not have a get_port_name attribute
                                sndr_output_node_proj_label = \
                                    f"{sndr_label}:{OutputPort.__name__}-{proj.sender.name}"
                                # rcvr_input_node_proj_label = \
                                #     f"{rcvr_label}:" \
                                #     f"{rcvr_input_node_proj_owner._get_port_name(rcvr_input_node_proj)}"
                        else:
                            rcvr_cim_proj_label = cim_label
                            sndr_output_node_proj_label = sndr_label

                        # Render Projection
                        _render_projection(enclosing_g,
                                           proj,
                                           sndr_output_node_proj_label,
                                           rcvr_cim_proj_label,
                                           proj_color)

                # Projections from input_CIM to INPUT nodes
                for output_port in composition.input_CIM.output_ports:
                    projs = output_port.efferents
                    for proj in projs:

                        proj_color = self.default_node_color
                        if proj not in composition_projections:
                            if not show_projections_not_in_composition:
                                continue
                            else:
                                proj_color=self.inactive_projection_color

                        # Get label for Node that receives the input (rcvr_label)
                        rcvr_input_node_proj = proj.receiver
                        if (isinstance(rcvr_input_node_proj.owner, CompositionInterfaceMechanism)
                                and show_nested is not NESTED):
                            rcvr_input_node_proj_owner = rcvr_input_node_proj.owner.composition
                        else:
                            rcvr_input_node_proj_owner = rcvr_input_node_proj.owner

                        if rcvr_input_node_proj_owner is composition.controller:
                            # Projections to contoller are handled under _assign_controller_components
                            continue

                        # Validate the Projection is to an INPUT node or a node that is shadowing one
                        if ((rcvr_input_node_proj_owner in composition_nodes and NodeRole.INPUT not in
                                self._get_roles_by_node(composition, rcvr_input_node_proj_owner, context))
                                and (proj.receiver.shadow_inputs in composition_nodes and NodeRole.INPUT not in
                                     self._get_roles_by_node(composition, proj.receiver.shadow_inputs, context))):
                            raise ShowGraphError(f"Projection from input_CIM of {composition.name} to node "
                                                   f"{rcvr_input_node_proj_owner} that is not an "
                                                   f"{NodeRole.INPUT.name} node or shadowing its "
                                                   f"{NodeRole.INPUT.name.lower()}.")

                        rcvr_label = self._get_graph_node_label(composition,
                                                           rcvr_input_node_proj_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for CIM's port as edge's sender
                            sndr_input_cim_proj_label = f"{cim_label}:{OutputPort.__name__}-{proj.sender.name}"
                            if (isinstance(rcvr_input_node_proj_owner, Composition)
                                    and show_nested is not NESTED):
                                rcvr_input_node_proj_label = rcvr_label
                            else:
                                # Need to use direct reference to proj.receiver rather than rcvr_input_node_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                port_type = type(proj.receiver)
                                rcvr_input_node_proj_label = \
                                    f"{rcvr_label}:{port_type.__name__}-{proj.receiver.name}"
                                # rcvr_input_node_proj_label = \
                                #     f"{rcvr_label}:" \
                                #     f"{rcvr_input_node_proj_owner._get_port_name(rcvr_input_node_proj)}"
                        else:
                            sndr_input_cim_proj_label = cim_label
                            rcvr_input_node_proj_label = rcvr_label

                        # Render Projection
                        _render_projection(g, proj, sndr_input_cim_proj_label, rcvr_input_node_proj_label, proj_color)

            # PARAMETER_CIM -------------------------------------------------------------------------

            elif cim is composition.parameter_CIM:

                # Projections from ControlMechanism(s) in enclosing Composition to parameter_CIM
                # (other than from controller;  that is handled in _assign_controller_compoents)
                for input_port in composition.parameter_CIM.input_ports:
                    projs = input_port.path_afferents
                    for proj in projs:

                        proj_color = self.control_color
                        if proj not in enclosing_comp_projections:
                            if not show_projections_not_in_composition:
                                continue
                            else:
                                proj_color=self.inactive_projection_color

                        # Get label for Node that sends the ControlProjection (sndr label)
                        ctl_mech_output_port = proj.sender
                        # Skip if sender is cim (handled by enclosing Composition's call to this method)
                        #   or Projections to cim aren't being shown (not NESTED)
                        if (isinstance(ctl_mech_output_port.owner, CompositionInterfaceMechanism)
                                or show_nested is not NESTED):
                            continue
                        else:
                            ctl_mech_output_port_owner = ctl_mech_output_port.owner
                        assert isinstance(ctl_mech_output_port_owner, ControlMechanism), \
                            f"PROGRAM ERROR: parameter_CIM of {composition.name} recieves a Projection " \
                            f"from a Node from other than a {ControlMechanism.__name__}."
                        # Skip Projections from controller (handled in _assign_controller_components)
                        if self._is_composition_controller(ctl_mech_output_port_owner, context, enclosing_comp):
                            continue
                        # Skip if there is no outer Composition (enclosing_g),
                        #    or Projections across nested Compositions are not being shown (show_nested=INSET)
                        if not enclosing_g or show_nested is INSET:
                            continue
                        sndr_label = self._get_graph_node_label(composition,
                                                           ctl_mech_output_port_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for ctl_mech's OutputPrt as edge's sender
                            sndr_ctl_sig_proj_label = \
                                f"{sndr_label}:{OutputPort.__name__}-{proj.sender.name}"
                            # Get label for CIM's InputPort as edge's receiver
                            port_type = type(proj.receiver)
                            rcvr_param_cim_proj_label = f"{cim_label}:{port_type.__name__}-{proj.receiver.name}"
                        else:
                            sndr_ctl_sig_proj_label = sndr_label
                            rcvr_param_cim_proj_label = cim_label

                        # Render Projection
                        _render_projection(enclosing_g, proj,
                                           sndr_ctl_sig_proj_label,
                                           rcvr_param_cim_proj_label,
                                           proj_color)

                # Projections from parameter_CIM to Nodes that are being modulated
                for output_port in composition.parameter_CIM.output_ports:
                    projs = output_port.efferents
                    for proj in projs:

                        proj_color = None
                        if proj not in composition_projections:
                            if not show_projections_not_in_composition:
                                continue
                            else:
                                proj_color=self.inactive_projection_color

                        # Get label for Node that receives modulation (modulated_mech_label)
                        rcvr_modulated_mech_proj = proj.receiver
                        if (isinstance(rcvr_modulated_mech_proj.owner, CompositionInterfaceMechanism)
                                and show_nested is not NESTED):
                            rcvr_modulated_mech_proj_owner = rcvr_modulated_mech_proj.owner.composition
                        else:
                            rcvr_modulated_mech_proj_owner = rcvr_modulated_mech_proj.owner

                        if rcvr_modulated_mech_proj_owner is composition.controller:
                            # Projections to contoller are handled under _assign_controller_components
                            # Note: at present controllers are not modulable; here for possible future condition(s)
                            continue
                        rcvr_label = self._get_graph_node_label(composition,
                                                                rcvr_modulated_mech_proj_owner,
                                                                show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for CIM's port as edge's sender
                            sndr_param_cim_proj_label = f"{cim_label}:{OutputPort.__name__}-{proj.sender.name}"
                            if (isinstance(rcvr_modulated_mech_proj_owner, Composition)
                                    and show_nested in [NESTED, INSET]):
                                rcvr_modulated_mec_proj_label = rcvr_label
                            else:
                                # Need to use direct reference to proj.receiver rather than rcvr_modulated_mech_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                port_type = type(proj.receiver)
                                rcvr_modulated_mec_proj_label = \
                                    f"{rcvr_label}:{port_type.__name__}-{proj.receiver.name}"
                                # rcvr_modulated_mec_proj_label = \
                                #     f"{rcvr_label}:" \
                                #     f"{rcvr_input_node_proj_owner._get_port_name(rcvr_modulated_mech_proj)}"
                        else:
                            sndr_param_cim_proj_label = cim_label
                            rcvr_modulated_mec_proj_label = rcvr_label

                        # Render Projection
                        if self._trace_senders_for_controller(proj, context, enclosing_comp):
                            ctl_proj_color = proj_color or self.controller_color
                        else:
                            ctl_proj_color = proj_color or self.control_color

                        if isinstance(proj, MappingProjection):
                            arrowhead = self.default_projection_arrow
                        else:
                            arrowhead = self.control_projection_arrow

                        if RANDOMIZATION_CONTROL_SIGNAL in proj.sender.name:
                            style = 'dashed'
                        else:
                            style = self.style

                        _render_projection(g, proj, sndr_param_cim_proj_label, rcvr_modulated_mec_proj_label,
                                           proj_color=ctl_proj_color, arrowhead=arrowhead, style=style)


            # OUTPUT_CIM ----------------------------------------------------------------------------

            elif cim is composition.output_CIM:

                # Projections from OUTPUT nodes to output_CIM
                for input_port in composition.output_CIM.input_ports:
                    projs = input_port.path_afferents
                    for proj in projs:

                        proj_color = self.default_node_color
                        if proj not in composition_projections:
                            if not show_projections_not_in_composition:
                                continue
                            else:
                                proj_color=self.inactive_projection_color
                        else:
                            port, node, comp = cim._get_source_info_from_output_CIM(proj.receiver)
                            if (node in comp.get_nodes_by_role(NodeRole.PROBE)
                                    and not composition.include_probes_in_output):
                                proj_color=self.probe_color

                        sndr_output_node_proj = proj.sender
                        if (isinstance(sndr_output_node_proj.owner, CompositionInterfaceMechanism)
                                and show_nested is not NESTED):
                            sndr_output_node_proj_owner = sndr_output_node_proj.owner.composition
                        else:
                            sndr_output_node_proj_owner = sndr_output_node_proj.owner
                        # Validate the Projection is from an OUTPUT or PROBE node
                        if (sndr_output_node_proj_owner in composition_nodes and
                                not any(role for role in {NodeRole.OUTPUT, NodeRole.PROBE}
                                        if role in self._get_roles_by_node(composition, sndr_output_node_proj_owner,
                                                                           context))):
                            raise ShowGraphError(f"Projection to output_CIM of {composition.name} "
                                                   f"from node {sndr_output_node_proj_owner} that is not "
                                                   f"an {NodeRole.OUTPUT} node.")

                        sndr_label = self._get_graph_node_label(composition,
                                                                sndr_output_node_proj_owner,
                                                                show_types, show_dimensions)

                        # Construct edge name
                        if show_node_structure:
                            # Get label of CIM's port as edge's receiver
                            port_type = type(proj.receiver)
                            rcvr_output_cim_proj_label = f"{cim_label}:{port_type.__name__}-{proj.receiver.name}"
                            if (isinstance(sndr_output_node_proj_owner, Composition)
                                    and show_nested is not NESTED):
                                sndr_output_node_proj_label = sndr_label
                            else:
                                # Need to use direct reference to proj.sender rather than sndr_output_node_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                sndr_output_node_proj_label = \
                                    f"{sndr_label}:{OutputPort.__name__}-{proj.sender.name}"
                                # sndr_output_node_proj_label = \
                                #     f"{sndr_label}:" \
                                #     f"{sndr_output_node_proj_owner._get_port_name(sndr_output_node_proj)}"
                        else:
                            sndr_output_node_proj_label = sndr_label
                            rcvr_output_cim_proj_label = cim_label

                        # FIX 6/23/20 PROBLEM POINT: (SEE MESSAGE FOR COMMIT eb61303808ad2a5ba46fdd18d0e583283397915c)
                        # Render Projection
                        _render_projection(g,
                                           proj,
                                           sndr_output_node_proj_label,
                                           rcvr_output_cim_proj_label,
                                           proj_color)

                # Projections from output_CIM to Node(s) in enclosing Composition
                for output_port in composition.output_CIM.output_ports:
                    projs = output_port.efferents
                    for proj in projs:

                        proj_color = self.default_node_color
                        if proj not in enclosing_comp_projections:
                            if not show_projections_not_in_composition:
                                continue
                            else:
                                proj_color=self.inactive_projection_color
                        rcvr_node_input_port = proj.receiver

                        # Skip if receiver is controller of enclosing_comp (handled by _assign_controller_components)
                        if rcvr_node_input_port.owner is enclosing_comp.controller:
                            continue

                        # Skip if receiver is cim (handled by enclosing Composition's call to this method)
                        if (isinstance(rcvr_node_input_port.owner, CompositionInterfaceMechanism) and
                                rcvr_node_input_port.owner.composition is enclosing_comp):
                            continue

                        # Skip if there is no inner Composition (show_nested!=NESTED) or
                        #   or Projections across nested Compositions are not being shown (show_nested=INSET)
                        if not enclosing_g or show_nested is INSET:
                            continue

                        # Skip if show_controller and the receiver is objective mechanism
                        if show_controller and enclosing_comp.controller \
                                and getattr(enclosing_comp.controller, 'objective_mechanism', None) \
                                is rcvr_node_input_port.owner:
                            continue

                        rcvr_node_input_port_owner = rcvr_node_input_port.owner

                        rcvr_label = self._get_graph_node_label(composition,
                                                                rcvr_node_input_port_owner,
                                                                show_types, show_dimensions)

                        # Construct edge name
                        if show_node_structure:
                            # Get label of CIM's port as edge's receiver
                            sndr_output_cim_proj_label = f"{cim_label}:{OutputPort.__name__}-{proj.sender.name}"
                            if (isinstance(rcvr_node_input_port_owner, Composition)
                                    and show_nested is not NESTED):
                                rcvr_input_node_proj_label = rcvr_label
                            else:
                                # Need to use direct reference to proj.sender rather than sndr_output_node_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                port_type = type(proj.receiver)
                                rcvr_input_node_proj_label = \
                                    f"{rcvr_label}:{port_type.__name__}-{proj.receiver.name}"
                                # rcvr_input_node_proj_label = \
                                #     f"{sndr_label}:" \
                                #     f"{sndr_output_node_proj_owner._get_port_name(sndr_output_node_proj)}"
                        else:
                            rcvr_input_node_proj_label = rcvr_label
                            sndr_output_cim_proj_label = cim_label

                        # Render Projection
                        _render_projection(enclosing_g,
                                           proj,
                                           sndr_output_cim_proj_label,
                                           rcvr_input_node_proj_label,
                                           proj_color)


    def _assign_controller_components(self,
                                      g,
                                      active_items,
                                      show_nested,
                                      show_cim,
                                      show_controller,
                                      show_learning,
                                      show_types,
                                      show_dimensions,
                                      show_node_structure,
                                      node_struct_args,
                                      show_projection_labels,
                                      show_projections_not_in_composition,
                                      comp_hierarchy,
                                      nesting_level,
                                      context):
        """Assign control nodes and edges to graph"""
        from psyneulink.core.compositions.composition import Composition

        composition = self.composition
        nodes = self._get_nodes(composition, context)
        controller = composition.controller

        if controller is None:
            # Only warn if there is no controller *and* no ControlProjections from an outer Composition
            if not composition.parameter_CIM.output_ports:
                warnings.warn(f"{composition.name} has not been assigned a \'controller\', "
                              f"so \'show_controller\' option in call to its show_graph() method will be ignored.")
            return

        # Assign colors, penwidth and label displayed for controller and ControlProjections ---------------------
        ctlr_color = self.controller_color
        if controller in active_items:
            if self.active_color != BOLD:
                ctlr_color = self.active_color
            ctlr_width = str(self.default_width + self.active_thicker_by)
            composition.active_item_rendered = True
        else:
            ctlr_color = self.controller_color
            ctlr_width = str(self.default_width)

        ctl_proj_color = self.controller_color
        if controller in active_items:
            if self.active_color != BOLD:
                ctl_proj_color = self.active_color
            ctl_proj_width = str(self.default_width + self.active_thicker_by)
            composition.active_item_rendered = True
        else:
            ctl_proj_width = str(self.default_width)

        # Assign controller node
        node_shape = self.mechanism_shape
        ctlr_label = self._get_graph_node_label(composition, controller, show_types, show_dimensions)
        if show_node_structure:
            g.node(ctlr_label,
                   controller._show_structure(**node_struct_args, node_border=ctlr_width,
                                              condition=composition.controller_condition),
                   shape=self.struct_shape,
                   color=ctlr_color,
                   penwidth=ctlr_width,
                   rank=self.control_rank
                   )
        else:
            g.node(ctlr_label,
                   color=ctlr_color, penwidth=ctlr_width, shape=self.controller_shape,
                   rank=self.control_rank)

        # outgoing edges (from controller to ProcessingMechanisms)
        for control_signal in controller.control_signals:
            for ctl_proj in control_signal.efferents:

                # # Allow MappingProjections to iconified rep of nested Composition to show as ControlProjection
                # if (isinstance(ctl_proj, ControlProjection)
                #         or (isinstance(ctl_proj.receiver.owner, CompositionInterfaceMechanism)
                #             and (not show_cim or not show_nested))):
                #     ctl_proj_arrowhead = self.control_projection_arrow
                # else:  # This is to expose an errant MappingProjection if one slips in
                #     ctl_proj_arrowhead = self.default_projection_arrow
                #     ctl_proj_color = self.default_node_color
                ctl_proj_arrowhead = self.control_projection_arrow

                # Skip ControlProjections not in the Composition
                if ctl_proj not in self._get_projections(composition, context):
                    continue

                # Construct edge name  ---------------------------------------------------

                # Get receiver label for ControlProjection as base for edge's receiver label
                # First get label for receiver's owner node (Mechanism or nested Composition), used below
                ctl_proj_rcvr = ctl_proj.receiver
                # If receiver is a parameter_CIM
                if isinstance(ctl_proj_rcvr.owner, CompositionInterfaceMechanism):
                    # Deal with ControlProjections across more than one level of nesting:
                    rcvr_comp = ctl_proj_rcvr.owner.composition
                    def find_rcvr_comp(r, c, l):
                        """Find deepest Composition within c that encloses r within range of num_nesting_levels of c"""
                        rcvr_nodes = self._get_nodes(c, context)
                        if (self.num_nesting_levels is not None and l > self.num_nesting_levels):
                            return c, l
                        elif r in rcvr_nodes:
                            return r, l
                        l+=1
                        for nested_c in [nc for nc in nodes if isinstance(nc, Composition)]:
                            return find_rcvr_comp(r, nested_c, l)
                        return None
                    project_to_node = False
                    try:
                        enclosing_comp, l = find_rcvr_comp(rcvr_comp, composition, 0)
                    except TypeError:
                        raise ShowGraphError(f"ControlProjection not found from {controller} in "
                                               f"{composition.name} to {rcvr_comp}")
                    if show_nested is NESTED:
                        # Node that receives ControlProjection is within num_nesting_levels, so show it
                        if self.num_nesting_levels is None or l < self.num_nesting_levels:
                            project_to_node = True
                        # Node is not within range, but its Composition is,
                        #     so leave rcvr_comp assigned to that, and don't project_to_node
                        elif l == self.num_nesting_levels:
                            pass
                        # Receiver's Composition is not within num_nesting_levels, so use closest one that encloses it
                        else:
                            rcvr_comp = enclosing_comp
                    else:
                        rcvr_comp = enclosing_comp

                    # if show_cim and show_nested is NESTED:
                    if show_cim and project_to_node:
                        # Use Composition's parameter_CIM port
                        ctl_proj_rcvr_owner = ctl_proj_rcvr.owner
                    elif project_to_node:
                        ctl_proj_rcvr = self._trace_receivers_for_terminal_receiver(ctl_proj_rcvr)
                        ctl_proj_rcvr_owner = ctl_proj_rcvr.owner
                    else:
                        # Use Composition if show_cim is False
                        ctl_proj_rcvr_owner = rcvr_comp
                # In all other cases, use Port (either ParameterPort of a Mech, or parameter_CIM for nested comp)
                else:
                    ctl_proj_rcvr_owner = ctl_proj_rcvr.owner

                rcvr_label = self._get_graph_node_label(composition, ctl_proj_rcvr_owner, show_types, show_dimensions)
                if (isinstance(ctl_proj_rcvr_owner, CompositionInterfaceMechanism)
                        or (isinstance(ctl_proj_rcvr_owner, Composition) and show_nested==INSET and show_cim)):
                    ctl_proj_arrowhead = self.default_projection_arrow

                # Get sender and receiver labels for edge
                if show_node_structure:
                    # Get label for controller's port as edge's sender
                    ctl_proj_sndr_label = ctlr_label + ':' + controller._get_port_name(control_signal)
                    # Get label for edge's receiver as owner Mechanism:
                    if (isinstance(ctl_proj_rcvr.owner, CompositionInterfaceMechanism) and show_nested is not NESTED):
                        ctl_proj_rcvr_label = rcvr_label
                    # Get label for edge's receiver as Port:
                    else:
                        ctl_proj_rcvr_label = rcvr_label + ':' + ctl_proj_rcvr_owner._get_port_name(ctl_proj_rcvr)
                else:
                    ctl_proj_sndr_label = ctlr_label
                    ctl_proj_rcvr_label = rcvr_label

                if show_projection_labels:
                    edge_label = ctl_proj.name
                else:
                    edge_label = ''

                if RANDOMIZATION_CONTROL_SIGNAL in ctl_proj.sender.name:
                    style = 'dashed'
                else:
                    style = self.style

                # Construct edge -----------------------------------------------------------------------
                g.edge(ctl_proj_sndr_label,
                       ctl_proj_rcvr_label,
                       label=edge_label,
                       color=ctl_proj_color,
                       penwidth=ctl_proj_width,
                       arrowhead=ctl_proj_arrowhead,
                       style = style
                       )

        # If controller has objective_mechanism, assign its node and Projections,
        #     including one from ObjectiveMechanism to controller
        if controller.objective_mechanism:
            # get projection from ObjectiveMechanism to ControlMechanism
            objmech_ctlr_proj = controller.input_port.path_afferents[0]
            if controller in active_items:
                if self.active_color == BOLD:
                    objmech_ctlr_proj_color = self.controller_color
                else:
                    objmech_ctlr_proj_color = self.active_color
                objmech_ctlr_proj_width = str(self.default_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                objmech_ctlr_proj_color = self.controller_color
                objmech_ctlr_proj_width = str(self.default_width)

            # get ObjectiveMechanism
            objmech = objmech_ctlr_proj.sender.owner
            if objmech in active_items:
                if self.active_color == BOLD:
                    objmech_color = self.controller_color
                else:
                    objmech_color = self.active_color
                objmech_width = str(self.default_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                objmech_color = self.controller_color
                objmech_width = str(self.default_width)

            objmech_label = self._get_graph_node_label(composition, objmech, show_types, show_dimensions)
            if show_node_structure:
                if objmech in composition.scheduler.conditions:
                    condition = composition.scheduler.conditions[objmech]
                else:
                    condition = None
                g.node(objmech_label,
                       objmech._show_structure(**node_struct_args, node_border=ctlr_width, condition=condition),
                       shape=self.struct_shape,
                       color=objmech_color,
                       penwidth=ctlr_width,
                       rank=self.control_rank
                       )
            else:
                g.node(objmech_label,
                        color=objmech_color, penwidth=objmech_width, shape=node_shape,
                        rank=self.control_rank)

            # objmech to controller edge
            if show_projection_labels:
                edge_label = objmech_ctlr_proj.name
            else:
                edge_label = ''
            if show_node_structure:
                obj_to_ctrl_label = objmech_label + ':' + objmech._get_port_name(objmech_ctlr_proj.sender)
                ctlr_from_obj_label = ctlr_label + ':' + objmech._get_port_name(objmech_ctlr_proj.receiver)
            else:
                obj_to_ctrl_label = objmech_label
                ctlr_from_obj_label = ctlr_label
            g.edge(obj_to_ctrl_label, ctlr_from_obj_label, label=edge_label,
                   color=objmech_ctlr_proj_color, penwidth=objmech_ctlr_proj_width)

            # incoming edges (from monitored mechs to objective mechanism)
            for input_port in objmech.input_ports:
                for projection in input_port.path_afferents:
                    # MODIFIED 1/6/22 NEW:
                    # Get nested source node for direct projection to objective mechanism
                    if isinstance(projection.sender.owner, CompositionInterfaceMechanism) and not show_cim:
                        cim_output_port = projection.sender
                        proj_sndr, node, comp = cim_output_port.owner._get_source_info_from_output_CIM(cim_output_port)
                    else:
                        proj_sndr = projection.sender
                    # MODIFIED 1/6/22 END
                    if objmech in active_items:
                        if self.active_color == BOLD:
                            proj_color = self.controller_color
                        else:
                            proj_color = self.active_color
                        proj_width = str(self.default_width + self.active_thicker_by)
                        composition.active_item_rendered = True
                    else:
                        proj_color = self.controller_color
                        proj_width = str(self.default_width)
                    if show_node_structure:
                        sndr_proj_label = self._get_graph_node_label(composition,
                                                                proj_sndr.owner,
                                                                show_types,
                                                                show_dimensions)
                        if (proj_sndr.owner not in nodes
                                # MODIFIED 1/6/22 NEW:
                                and isinstance(proj_sndr.owner, CompositionInterfaceMechanism)):
                            # MODIFIED 1/6/22 END
                            num_nesting_levels = self.num_nesting_levels or 0
                            nested_comp = proj_sndr.owner.composition
                            try:
                                nesting_depth = next((k for k, v in comp_hierarchy.items() if v == nested_comp))
                                sender_visible = nesting_depth <= num_nesting_levels
                            except StopIteration:
                                sender_visible = False
                        else:
                            sender_visible = True
                        if sender_visible:
                            sndr_proj_label += ':' + objmech._get_port_name(proj_sndr)
                        objmech_proj_label = objmech_label + ':' + objmech._get_port_name(input_port)
                    else:
                        sndr_proj_label = self._get_graph_node_label(composition,
                                                                proj_sndr.owner,
                                                                show_types,
                                                                show_dimensions)
                        objmech_proj_label = self._get_graph_node_label(composition,
                                                                   objmech,
                                                                   show_types,
                                                                   show_dimensions)
                    if show_projection_labels:
                        edge_label = projection.name
                    else:
                        edge_label = ''
                    g.edge(sndr_proj_label, objmech_proj_label, label=edge_label,
                           color=proj_color, penwidth=proj_width)

        # If controller has no objective_mechanism but does have outcome_input_ports, add Projections from them
        elif controller.num_outcome_input_ports:
            # incoming edges (from monitored mechs directly to controller)
            for outcome_input_port in controller.outcome_input_ports:
                for projection in outcome_input_port.path_afferents:
                    # MODIFIED 1/6/22 NEW:
                    # Handled by _assign_cim_components()
                    if isinstance(projection.sender.owner, CompositionInterfaceMechanism) and not show_cim:
                        continue
                    # MODIFIED 1/6/22 END
                    if show_node_structure and show_cim:
                        sndr_proj_label = self._get_graph_node_label(composition,
                                                                     projection.sender.owner,
                                                                     show_types,
                                                                     show_dimensions)
                        if (projection.sender.owner not in nodes
                                and not controller.allow_probes):
                            num_nesting_levels = self.num_nesting_levels or 0
                            nested_comp = projection.sender.owner.composition
                            try:
                                nesting_depth = next((k for k, v in comp_hierarchy.items() if v == nested_comp))
                                sender_visible = nesting_depth <= num_nesting_levels
                            except StopIteration:
                                sender_visible = False
                        else:
                            sender_visible = True
                        if sender_visible:
                            sndr_proj_label += ':' + controller._get_port_name(projection.sender)
                        ctlr_input_proj_label = ctlr_label + ':' + controller._get_port_name(outcome_input_port)
                    else:
                        sndr_proj_label = self._get_graph_node_label(composition,
                                                                projection.sender.owner,
                                                                show_types,
                                                                show_dimensions)
                        ctlr_input_proj_label = self._get_graph_node_label(composition,
                                                                   controller,
                                                                   show_types,
                                                                   show_dimensions)
                    if show_projection_labels:
                        edge_label = projection.name
                    else:
                        edge_label = ''
                    g.edge(sndr_proj_label, ctlr_input_proj_label, label=edge_label,
                           # color=proj_color, penwidth=proj_width)
                           color=ctl_proj_color, penwidth=ctl_proj_width)

        # If controller has an agent_rep, assign its node and edges (not Projections per se)
        if hasattr(controller, 'agent_rep') and controller.agent_rep and show_controller==AGENT_REP :
            # get agent_rep
            agent_rep = controller.agent_rep
            # controller is active, treat
            if controller in active_items:
                if self.active_color == BOLD:
                    agent_rep_color = self.controller_color
                else:
                    agent_rep_color = self.active_color
                agent_rep_width = str(self.default_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                agent_rep_color = self.controller_color
                agent_rep_width = str(self.default_width)

            # agent_rep node
            agent_rep_label = self._get_graph_node_label(composition, agent_rep, show_types, show_dimensions)
            g.node(agent_rep_label,
                    color=agent_rep_color, penwidth=agent_rep_width, shape=self.agent_rep_shape,
                    rank=self.control_rank)

            # agent_rep <-> controller edges
            g.edge(agent_rep_label, ctlr_label, color=agent_rep_color, penwidth=agent_rep_width)
            g.edge(ctlr_label, agent_rep_label, color=agent_rep_color, penwidth=agent_rep_width)

        # get any state_feature projections and any other incoming edges to controller
        senders = set()
        # FIX: 11/3/21 - NEED TO MODIFY ONCE OUTCOME InputPorts ARE MOVED
        for i in controller.input_ports[controller.num_outcome_input_ports:]:
            for p in i.path_afferents:
                # MODIFIED 1/6/22 NEW:
                sender = p.sender.owner
                if isinstance(sender, CompositionInterfaceMechanism) and not show_cim:
                    pass # FIX: 1/6/22 - PLACEMARKER FOR RELABELING INPUT_CIM AS SHADOWING INPUT OF SHADOWED NODE
                    assert True
                # MODIFIED 1/6/22 END
                senders.add(sender)
        self._assign_incoming_edges(g,
                                    controller,
                                    ctlr_label,
                                    senders,
                                    active_items,
                                    show_nested,
                                    show_cim,
                                    show_learning,
                                    show_types,
                                    show_dimensions,
                                    show_node_structure,
                                    show_projection_labels,
                                    show_projections_not_in_composition,
                                    proj_color=ctl_proj_color,
                                    comp_hierarchy=comp_hierarchy,
                                    nesting_level=nesting_level,
                                    context=context)

    def _assign_learning_components(self,
                                    g,
                                    processing_graph,
                                    enclosing_comp,
                                    comp_hierarchy,
                                    nesting_level,
                                    active_items,
                                    show_nested,
                                    show_cim,
                                    show_learning,
                                    show_types,
                                    show_dimensions,
                                    show_node_structure,
                                    node_struct_args,
                                    show_projection_labels,
                                    show_projections_not_in_composition,
                                    context):
        """Assign learning nodes and edges to graph"""

        from psyneulink.core.compositions.composition import NodeRole
        composition = self.composition

        # Get learning_components, with exception of INPUT (i.e. TARGET) nodes
        #    (i.e., allow TARGET node to continue to be marked as an INPUT node)
        learning_components = [node for node in composition.learning_components]
        # learning_components.extend([node for node in composition.nodes if
        #                             NodeRole.AUTOASSOCIATIVE_LEARNING in
        #                             composition.nodes_to_roles[node]])

        for rcvr in learning_components:
            # if rcvr is Projection, skip (handled in _assign_processing_components)
            if isinstance(rcvr, MappingProjection):
                return

            if NodeRole.TARGET in self._get_roles_by_node(composition, rcvr, context):
                rcvr_width = self.bold_width
            else:
                rcvr_width = self.default_width

            # Get rcvr info
            learning_rcvr_label = self._get_graph_node_label(composition,
                                                             rcvr,
                                                             show_types, show_dimensions)
            if rcvr in active_items:
                if self.active_color == BOLD:
                    rcvr_color = self.learning_color
                else:
                    rcvr_color = self.active_color
                rcvr_width = str(rcvr_width + self.active_thicker_by)
                composition.active_item_rendered = True
            else:
                rcvr_color = self.learning_color
                rcvr_width = str(rcvr_width)

            # rcvr is a LearningMechanism or ObjectiveMechanism (ComparatorMechanism)
            # Implement node for Mechanism
            if show_node_structure:
                g.node(learning_rcvr_label,
                       rcvr._show_structure(**node_struct_args,
                                            node_border=rcvr_width),
                       shape=self.struct_shape,
                       color=rcvr_color,
                       penwidth=rcvr_width,
                       rank=self.learning_rank)
            else:
                g.node(learning_rcvr_label,
                       shape=self.mechanism_shape,
                       color=rcvr_color,
                       penwidth=rcvr_width,
                       rank=self.learning_rank)

            # Implement sender edges
            if show_learning is not SHOW_JUST_LEARNING_PROJECTIONS:
                sndrs = processing_graph[rcvr]
                self._assign_incoming_edges(g,
                                            rcvr,
                                            learning_rcvr_label,
                                            sndrs,
                                            active_items,
                                            show_nested,
                                            show_cim,
                                            show_learning,
                                            show_types,
                                            show_dimensions,
                                            show_node_structure,
                                            show_projection_labels,
                                            show_projections_not_in_composition,
                                            enclosing_comp=enclosing_comp,
                                            comp_hierarchy=comp_hierarchy,
                                            nesting_level=nesting_level,
                                            context=context)

    def _render_projection_as_node(self,
                                   g,
                                   active_items,
                                   show_node_structure,
                                   show_learning,
                                   show_types,
                                   show_dimensions,
                                   show_projection_labels,
                                   show_projections_not_in_composition,
                                   proj,
                                   label,
                                   proj_color,
                                   proj_width,
                                   sndr_label=None,
                                   rcvr_label=None,
                                   context=None):

        composition = self.composition

        proj_receiver = proj.receiver.owner

        # Node for Projection
        g.node(label, shape=self.learning_projection_shape, color=proj_color, penwidth=proj_width)

        if proj_receiver in active_items:
            if self.active_color == BOLD:
                edge_color = proj_color
            else:
                edge_color = self.active_color
            edge_width = str(self.default_width + self.active_thicker_by)
        else:
            edge_color = self.default_node_color
            edge_width = str(self.default_width)

        # Edges to and from Projection node
        if sndr_label:
            self.G.edge(sndr_label, label, arrowhead='none',
                   color=edge_color, penwidth=edge_width)
        if rcvr_label:
            self.G.edge(label, rcvr_label,
                   color=edge_color, penwidth=edge_width)

        # LearningProjection(s) to node
        # if proj in active_items or (proj_learning_in_execution_phase and proj_receiver in active_items):
        if show_learning is LEARNABLE:
            if not proj.learnable:
                return True

        if proj in active_items:
            if self.active_color == BOLD:
                learning_proj_color = self.learning_color
            else:
                learning_proj_color = self.active_color
            learning_proj_width = str(self.default_width + self.active_thicker_by)
            composition.active_item_rendered = True
        else:
            learning_proj_color = self.learning_color
            learning_proj_width = str(self.default_width)
        sndrs = proj._parameter_ports['matrix'].mod_afferents # GET ALL LearningProjections to proj
        for sndr in sndrs:
            sndr_label = self._get_graph_node_label(composition, sndr.sender.owner, show_types, show_dimensions)
            rcvr_label = self._get_graph_node_label(composition, proj, show_types, show_dimensions)
            if show_projection_labels:
                edge_label = proj._parameter_ports['matrix'].mod_afferents[0].name
            else:
                edge_label = ''
            if show_node_structure:
                # self.G.edge(sndr_label + ':' + OutputPort.__name__ + '-' + 'LearningSignal',
                #        rcvr_label,
                #        label=edge_label,
                #        color=learning_proj_color, penwidth=learning_proj_width)
                self._implement_graph_edge(self.G, proj, context,
                                           sndr_label + ':' + OutputPort.__name__ + '-' + 'LearningSignal',
                                           rcvr_label,
                                           label=edge_label,
                                           color=learning_proj_color,
                                           penwidth=learning_proj_width)
            else:
                # self.G.edge(sndr_label, rcvr_label, label = edge_label,
                #        color=learning_proj_color, penwidth=learning_proj_width)
                self._implement_graph_edge(self.G, proj, context,
                                           sndr_label,
                                           rcvr_label,
                                           label=edge_label,
                                           color=learning_proj_color,
                                           penwidth=learning_proj_width)
        return True

    @beartype
    def _assign_incoming_edges(self,
                               g,
                               rcvr,
                               rcvr_label,
                               senders,
                               active_items,
                               show_nested,
                               show_cim,
                               show_learning,
                               show_types,
                               show_dimensions,
                               show_node_structure,
                               show_projection_labels,
                               show_projections_not_in_composition,
                               proj_color=None,
                               proj_arrow=None,
                               enclosing_comp=None,
                               comp_hierarchy=None,
                               nesting_level=None,
                               context=None):

        from psyneulink.core.compositions.composition import Composition, NodeRole
        composition = self.composition
        composition_projections = self._get_projections(composition, context)
        if nesting_level not in comp_hierarchy:
            comp_hierarchy[nesting_level] = composition
        enclosing_g = enclosing_comp._show_graph.G if enclosing_comp else None

        proj_color_default = proj_color or self.default_node_color
        proj_arrow_default = proj_arrow or self.default_projection_arrow

        # Deal with Projections from outer (enclosing_g) and inner (nested) Compositions
        # If not showing CIMs, then set up to find node for sender in inner or outer Composition
        if not show_cim:
            # Get sender node from inner Composition
            if show_nested is NESTED:
                # Add output_CIMs for nested Comps to find sender nodes
                cims = set([proj.sender.owner for proj in rcvr.afferents
                            if (proj in composition_projections
                                and isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                                and (proj.sender.owner is proj.sender.owner.composition.output_CIM))])
                senders.update(cims)
            # Get sender Node from outer Composition (enclosing_g)
            if enclosing_g and show_nested is not INSET:
                # Add input_CIM for current Composition to find senders from enclosing_g
                cims = set([proj.sender.owner for proj in rcvr.afferents
                            if (proj in composition_projections
                                and isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                                and proj.sender.owner in {composition.input_CIM, composition.parameter_CIM})])
                senders.update(cims)
            # HACK: FIX 6/13/20 - ADD USER-SPECIFIED TARGET NODE FOR INNER COMPOSITION (NOT IN processing_graph)

        def assign_sender_edge(sndr:Union[Mechanism, Composition],
                               proj:Projection,
                               proj_color:str,
                               proj_arrowhead:str
                               ) -> None:
            """Assign edge from sender to rcvr"""

            # Set sndr info
            sndr_label = self._get_graph_node_label(composition, sndr, show_types, show_dimensions)

            # Skip any projections to ObjectiveMechanism for controller
            #   (those are handled in _assign_controller_components)
            if (composition.controller and
                    proj.receiver.owner in {composition.controller.objective_mechanism}):
                return

            # Only consider Projections to the rcvr (or its CIM if rcvr is a Composition)
            if ((isinstance(rcvr, (Mechanism, Projection)) and proj.receiver.owner == rcvr)
                    or (isinstance(rcvr, Composition)
                        and proj.receiver.owner in {rcvr.input_CIM,
                                                    rcvr.parameter_CIM})):
                if show_node_structure and isinstance(sndr, Mechanism):
                    # If proj is a ControlProjection that comes from a parameter_CIM, get the port for the sender
                    if (isinstance(proj, ControlProjection) and
                            isinstance(proj.sender.owner, CompositionInterfaceMechanism)):
                        sndr_port = sndr.output_port
                    # Usual case: get port from Projection's sender
                    else:
                        sndr_port = proj.sender
                    sndr_port_owner = sndr_port.owner
                    if isinstance(sndr_port_owner, CompositionInterfaceMechanism) and rcvr is not composition.controller:
                        # Sender is input_CIM or parameter_CIM
                        if sndr_port_owner in {sndr_port_owner.composition.input_CIM,
                                               sndr_port_owner.composition.parameter_CIM}:
                            # Get port for node of outer Composition that projects to it
                            sndr_port = [v[0] for k,v in sender.port_map.items()
                                         if k is proj.receiver][0].path_afferents[0].sender
                        # Sender is output_CIM
                        else:
                            # Get port for node of inner Composition that projects to it
                            sndr_port = [k for k,v in sender.port_map.items() if v[1] is proj.sender][0]
                    sndr_proj_label = f'{sndr_label}:{sndr._get_port_name(sndr_port)}'
                else:
                    sndr_proj_label = sndr_label
                if show_node_structure and isinstance(rcvr, Mechanism):
                    proc_mech_rcvr_label = f'{rcvr_label}:{rcvr._get_port_name(proj.receiver)}'
                else:
                    proc_mech_rcvr_label = rcvr_label

                try:
                    has_learning = proj.has_learning_projection is not None
                except AttributeError:
                    has_learning = None

                edge_label = self._get_graph_node_label(composition, proj, show_types, show_dimensions)
                is_learning_component = (rcvr in composition.learning_components
                                         or sndr in composition.learning_components)
                if isinstance(sender, ControlMechanism):
                    proj_color = self.control_color
                    if (not isinstance(rcvr, Composition)
                            or (not show_cim and
                                (show_nested is not NESTED)
                                or (show_nested is False))):
                        # # Allow MappingProjections to iconified rep of nested Composition to show as ControlProjection
                        # if (isinstance(proj, ControlProjection)
                        #         or (isinstance(proj.receiver.owner, CompositionInterfaceMechanism)
                        #             and (not show_cim or not show_nested))):
                        #     proj_arrowhead = self.control_projection_arrow
                        # else:  # This is to expose an errant MappingProjection if one slips in
                        #     proj_arrowhead = self.default_projection_arrow
                        #     proj_color = self.default_node_color
                        proj_arrowhead = self.control_projection_arrow
                # Check if Projection or its receiver is active
                if any(item in active_items for item in {proj, proj.receiver.owner}):
                    if self.active_color == BOLD:
                        # if (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
                        if is_learning_component:
                            proj_color = self.learning_color
                        else:
                            pass
                    else:
                        proj_color = self.active_color
                    proj_width = str(self.default_width + self.active_thicker_by)
                    composition.active_item_rendered = True

                # Projection to or from a LearningMechanism
                elif (NodeRole.LEARNING in self._get_roles_by_node(composition, rcvr, context)):
                    proj_color = self.learning_color
                    proj_width = str(self.default_width)

                else:
                    proj_width = str(self.default_width)
                proc_mech_label = edge_label

                # RENDER PROJECTION AS EDGE

                if show_learning and has_learning:
                    # Render Projection as Node
                    #    (do it here rather than in _assign_learning_components,
                    #     as it needs afferent and efferent edges to other nodes)
                    # IMPLEMENTATION NOTE: Projections can't yet use structured nodes:
                    deferred = not self._render_projection_as_node(g,
                                                                   active_items,
                                                                   show_node_structure,
                                                                   show_learning,
                                                                   show_types,
                                                                   show_dimensions,
                                                                   show_projection_labels,
                                                                   show_projections_not_in_composition,
                                                                   proj,
                                                                   label=proc_mech_label,
                                                                   rcvr_label=proc_mech_rcvr_label,
                                                                   sndr_label=sndr_proj_label,
                                                                   proj_color=proj_color,
                                                                   proj_width=proj_width,
                                                                   context=context)
                    # Deferred if it is the last Mechanism in a learning Pathway
                    # (see _render_projection_as_node)
                    if deferred:
                        return

                else:
                    # Render Projection as edge
                    if show_projection_labels:
                        label = proc_mech_label
                    else:
                        label = ''

                    if assign_proj_to_enclosing_comp:
                        graph = enclosing_g
                    else:
                        graph = g

                    self._implement_graph_edge(graph,
                                               proj,
                                               context,
                                               sndr_proj_label,
                                               proc_mech_rcvr_label,
                                               label=label,
                                               color=proj_color,
                                               penwidth=proj_width,
                                               arrowhead=proj_arrowhead)

        # Sorted to insure consistency of ordering in g for testing
        for sender in sorted(senders):

            # Remove any Compositions from sndrs if show_cim is False and show_nested is True
            #    (since in that case the nodes for Compositions are bypassed)
            if not show_cim and show_nested is NESTED and isinstance(sender, Composition):
                continue

            # Iterate through all Projections from all OutputPorts of sender
            for output_port in sender.output_ports:
                for proj in output_port.efferents:

                    proj_color = proj_color_default
                    proj_arrowhead = proj_arrow_default

                    if not self._proj_in_composition(proj, composition_projections, context):
                        if show_projections_not_in_composition:
                            proj_color=self.inactive_projection_color
                        else:
                            continue

                    assign_proj_to_enclosing_comp = False

                    # Skip if sender is Composition and Projections to and from cim are being shown
                    #    (show_cim and show_nested) -- handled by _assign_cim_components
                    if isinstance(sender, Composition) and show_cim and show_nested is NESTED:
                        continue

                    if isinstance(sender, CompositionInterfaceMechanism):

                        # sender is input_CIM or parameter_CIM
                        if sender in {composition.input_CIM, composition.parameter_CIM}:
                            # FIX 6/2/20:
                            #     DELETE ONCE FILTERED BASED ON nesting_level IS IMPLEMENTED BEFORE CALL TO METHOD
                            # If cim has no afferents, presumably it is for the outermost Composition,
                            #     and therefore is not passing an afferent Projection from that Composition
                            if not sender.afferents and rcvr is not composition.controller:
                                continue
                            # FIX: LOOP HERE OVER sndr_spec IF THERE ARE SEVERAL
                            # Get node(s) from enclosing Composition that is/are source(s) of sender(s)
                            sndrs_specs = self._trace_senders_for_original_sender_mechanism(proj, nesting_level)
                            if not sndrs_specs:
                                continue
                            for sndr_spec in sorted(sndrs_specs):
                                sndr, sndr_port, sndr_nesting_level = sndr_spec
                                # if original sender is more than one level above receiver, replace enclosing_g with
                                # the g of the original sender composition
                                enclosing_comp = comp_hierarchy[sndr_nesting_level]
                                enclosing_g = enclosing_comp._show_graph.G
                                # Skip:
                                # - cims as sources (handled in _assign_cim_components)
                                #    unless it is the input_CIM for the outermost Composition and show_cim is not true
                                # - controller (handled in _assign_controller_components)
                                if (isinstance(sndr, CompositionInterfaceMechanism) and
                                        rcvr is not enclosing_comp.controller
                                        and rcvr is not composition.controller
                                        and not sndr.afferents and show_cim
                                        or self._is_composition_controller(sndr, context, enclosing_comp)):
                                    continue
                                if sender is composition.parameter_CIM:
                                    # # Allow MappingProjections to iconified rep of nested Composition
                                    # # to show as ControlProjection
                                    # if (isinstance(proj, ControlProjection)
                                    #         or (isinstance(proj.receiver.owner, CompositionInterfaceMechanism)
                                    #             and (not show_cim or not show_nested))):
                                    #     proj_arrowhead = self.control_projection_arrow
                                    #     proj_color = self.control_color
                                    # else:   # This is to expose an errant MappingProjection if one slips in
                                    #     proj_arrowhead = self.default_projection_arrow
                                    #     proj_color = proj_color=self.default_node_color
                                    proj_color = self.control_color
                                    proj_arrowhead = self.control_projection_arrow
                                assign_proj_to_enclosing_comp = True
                                assign_sender_edge(sndr, proj, proj_color, proj_arrowhead)
                            continue

                        # sender is output_CIM
                        # FIX: 4/5/21 IMPLEMENT LOOP HERE COMPARABLE TO ONE FOR input_CIM ABOVE
                        else:
                            # FIX 6/2/20:
                            #     DELETE ONCE FILTERED BASED ON nesting_level IS IMPLEMENTED BEFORE CALL TO METHOD
                            if not sender.efferents:
                                continue
                            # Insure cim has only one afferent
                            assert len([k.owner for k,v in sender.port_map.items() if v[1] is proj.sender])==1, \
                                f"PROGRAM ERROR: {sender} of {composition.name} has more than one efferent Projection."
                            # Get Node from nested Composition that projects to rcvr
                            sndr = [k.owner for k,v in sender.port_map.items() if v[1] is proj.sender][0]
                            # Skip:
                            # - cims as sources (handled in _assign_cim_components)
                            # - controller (handled in _assign_controller_components)
                            # NOTE 7/20/20: if receiver is a controller, then we need to skip this block or shadow inputs
                            # is not rendered -DS
                            if (rcvr is not composition.controller
                                    and isinstance(sndr, CompositionInterfaceMechanism)
                                    or (isinstance(sndr, ControlMechanism) and sndr.composition)):
                                continue
                    else:
                        sndr = sender

                    assign_sender_edge(sndr, proj, proj_color, proj_arrowhead)

    def _generate_output(self,
                         G,
                         enclosing_comp,
                         active_items,
                         show_controller,
                         output_fmt,
                         context
                         ):

        from psyneulink.core.compositions.composition import Composition, NodeRole

        composition = self.composition
        nodes = self._get_nodes(composition, context)
        projections = self._get_projections(composition, context)

        # Sort nodes for display
        def get_index_of_node_in_G_body(node, node_type: Literal['MECHANISM', 'Projection', 'Composition']):
            """Get index of node in G.body"""
            for i, item in enumerate(G.body):
                quoted_items = item.split('"')[1::2]
                if ((quoted_items and node.name == quoted_items[0])
                        or (node.name + ' [' in item)) and node_type in {MECHANISM, PROJECTION}:
                    if node_type in {MECHANISM}:
                        if '->' not in item:
                            return i
                    elif node_type in {PROJECTION}:
                        if '->' in item:
                            return i
                    else:
                        assert False, f'PROGRAM ERROR: node ({node.name}) not Mechanism or Projection in G.body'
                elif 'subgraph' in item and node_type in {COMPOSITION}:
                    return i

        for node in nodes:
            if isinstance(node, Composition):
                continue
            roles = self._get_roles_by_node(composition, node, context)
            # Put INPUT node(s) first
            if NodeRole.INPUT in roles:
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(0,G.body.pop(i))
            # Put OUTPUT node(s) last, except for controller of Composition and nested Compositions (see below)
            if NodeRole.OUTPUT in roles:
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))
            # Put ControlMechanism(s) last except for nested Compositions (see below)
            if isinstance(node, ControlMechanism):
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))

        for proj in projections:
            # Put ControlProjection(s) last, except for controller of Composition (see below)
            # if isinstance(proj, ControlProjection) and self._is_composition_controller(proj.sender.owner):
            if isinstance(proj, ControlProjection) and self._is_composition_controller(proj.sender.owner, context,
                                                                                       enclosing_comp):
                i = get_index_of_node_in_G_body(proj, PROJECTION)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))

        # Put controller of Composition, except for nested Composition(s)
        if composition.controller and show_controller:
            i = get_index_of_node_in_G_body(composition.controller, MECHANISM)
            G.body.insert(len(G.body),G.body.pop(i))

        # Put nested Composition(s) very last
        for node in nodes:
            if isinstance(node, Composition):
                i = get_index_of_node_in_G_body(node, COMPOSITION)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))
                    while (G.body[i][0:2] != "\t}"):
                        G.body.insert(len(G.body),G.body.pop(i))
                    G.body.insert(len(G.body),G.body.pop(i))


        # GENERATE OUTPUT ---------------------------------------------------------------------

        # Show as pdf
        try:
            if output_fmt == 'pdf':
                # G.format = 'svg'
                G.view(composition.name.replace(" ", "-"), cleanup=True, directory=get_default_showgraph_dir().joinpath('PDFS'))

            # Generate images for animation
            elif output_fmt == 'gif':
                if composition.active_item_rendered or INITIAL_FRAME in active_items:
                    self._generate_gifs(G, active_items, context)

            # Return graph to show in jupyter
            elif output_fmt == 'jupyter':
                return G

            elif output_fmt == 'gv':
                return G

            elif output_fmt == 'source':
                return G.source

            elif not output_fmt:
                return None

            else:
                raise ShowGraphError(f"Bad arg in call to {composition.name}.show_graph: '{output_fmt}'.")

        except ShowGraphError as e:
            # raise ShowGraphError(str(e.error_value))
            raise ShowGraphError(str(e.error_value)) from e
        # except:
        #     raise ShowGraphError(f"Problem displaying graph for {composition.name}")
        except Exception as e:
            raise ShowGraphError(f"Problem displaying graph for {composition.name}: {e}") from e

    def _is_composition_controller(self, mech, context, enclosing_comp=None):
        # FIX 6/12/20: REPLACE WITH TEST FOR NodeRole.CONTROLLER ONCE THAT IS IMPLEMENTED
        # return isinstance(mech, ControlMechanism) and hasattr(mech, 'composition') and mech.composition
        from psyneulink.core.compositions.composition import NodeRole
        if not isinstance(mech, ControlMechanism):
            return False
        for comp in [self.composition, enclosing_comp]:
            if not comp:
                continue
            if mech in comp._all_nodes and NodeRole.CONTROLLER in self._get_roles_by_node(comp, mech, context):
                return True
        return False

    def _trace_senders_for_controller(self, proj, context, comp=None):
        """Check whether source sender of a ControlProjection is (at any level of nesting) a Composition controller."""
        owner = proj.sender.owner
        comp = owner.composition if hasattr(owner, 'composition') else comp or self.composition
        if self._is_composition_controller(owner, context, comp):
            return True
        if isinstance(owner, CompositionInterfaceMechanism):
            sender_proj = owner.port_map[proj.receiver][0].path_afferents[0]
            return self._trace_senders_for_controller(sender_proj, comp)
        return False

    def _trace_senders_for_original_sender_mechanism(self,
                                                     proj:Projection,
                                                     nesting_level:int,
                                                     comp=None
                                                     ) -> [(Mechanism, Port, int)]:
        """
        Find the original sender of a projection that is routed through n-levels of cims.
        If there is no outer root Mechanism, as in the case of a nested input Mechanism that is not projected to,
        return None.
        """
        sender = proj.sender
        owner = sender.owner
        # Insure relevant InputPort of cim has only one afferent
        if isinstance(owner, CompositionInterfaceMechanism) and proj.receiver in owner.port_map:
            nesting_level -= 1
            num_afferents = len(owner.port_map[proj.receiver][0].path_afferents)
            if num_afferents == 0:
                # MODIFIED 1/6/22 OLD:
                return None
                # # MODIFIED 1/6/22 NEW:
                # # Presumably outermost Composition, so return CIM itself
                # return [(owner, sender, nesting_level)]
                # MODIFIED 1/6/22 END
            # # FIX: ITERATE OVER ALL AFFERENTS TO relevant InputPort of cim:
            # # MODIFIED 4/5/21 OLD:
            # outer_proj = owner.port_map[proj.receiver][0].path_afferents[0]
            # enclosing_showgraph = owner.composition._show_graph
            # return enclosing_showgraph._trace_senders_for_original_sender_mechanism(outer_proj, nesting_level)
            # MODIFIED 4/5/21 NEW:
            senders = []
            for outer_proj in owner.port_map[proj.receiver][0].path_afferents:
                enclosing_showgraph = owner.composition._show_graph
                sndrs = enclosing_showgraph._trace_senders_for_original_sender_mechanism(outer_proj, nesting_level)
                if sndrs is not None:
                    senders.extend(sndrs)
                # MODIFIED 1/6/22 NEW:
                else:
                    senders.append((outer_proj.sender.owner, sender, nesting_level))
                # MODIFIED 1/6/22 END
            return senders
            # MODIFIED 4/5/21 END
        # FIX: RECEIVERS OF THIS RETURN NEED TO HANDLE LIST
        return [(owner, sender, nesting_level)]

    def _trace_receivers_for_terminal_receiver(self, receiver, comp=None):
        """Find the terminal ParameterPort for a given intermediary input port of a parameter CIM"""
        parameter_port_map = receiver.owner.composition.parameter_CIM_ports
        ctl_proj_rcvr = next((k for k, v in parameter_port_map.items()
                              if parameter_port_map[k][0] is receiver), None)
        # hack needed for situation where multiple external control projections project to a single parameter port.
        # in this case, the receiver is overwritten in the parameter_CIM_ports dict.
        if ctl_proj_rcvr is None:
            pcim_input_ports = receiver.owner.composition.parameter_CIM.input_ports
            pcim_output_ports = receiver.owner.composition.parameter_CIM.output_ports
            for idx in range(len(pcim_input_ports)):
                if pcim_input_ports[idx] is receiver:
                    ctl_proj_rcvr = pcim_output_ports[idx].efferents[0].receiver
        if isinstance(ctl_proj_rcvr.owner, CompositionInterfaceMechanism):
            return receiver.owner.composition._show_graph._trace_receivers_for_terminal_receiver(ctl_proj_rcvr)
        return ctl_proj_rcvr

    def _get_graph_node_label(self, composition, item, show_types=None, show_dimensions=None):

        from psyneulink.core.compositions.composition import Composition
        if not isinstance(item, (Mechanism, Composition, Projection)):
            raise ShowGraphError(f"Unrecognized node type ({item}) in graph for {composition.name}.")

        name = item.name

        if show_types:
            name = item.name + '\n(' + item.__class__.__name__ + ')'

        if show_dimensions in {ALL, MECHANISMS} and isinstance(item, Mechanism):
            input_str = "in ({})".format(",".join(str(input_port.socket_width)
                                                  for input_port in item.input_ports))
            output_str = "out ({})".format(",".join(str(len(np.atleast_1d(output_port.value)))
                                                    for output_port in item.output_ports))
            return f"{output_str}\n{name}\n{input_str}"
        if show_dimensions in {ALL, PROJECTIONS} and isinstance(item, Projection):
            # MappingProjections use matrix
            if isinstance(item, MappingProjection):
                value = np.array(item.matrix)
                dim_string = "({})".format("x".join([str(i) for i in value.shape]))
                return "{}\n{}".format(item.name, dim_string)
            # ModulatoryProjections use value
            else:
                value = np.array(item.value)
                dim_string = "({})".format(len(value))
                return "{}\n{}".format(item.name, dim_string)

        if isinstance(item, CompositionInterfaceMechanism):
            name = name.replace('Input_CIM','INPUT_CIM')
            name = name.replace('Parameter_CIM', 'PARAMETER_CIM')
            name = name.replace('Output_CIM', 'OUTPUT_CIM')

        return name

    def _set_up_animation(self, context):

        composition = self.composition

        composition._component_animation_execution_count = None

        if isinstance(composition._animate, dict):
            # Assign directory for animation files
            default_dir = get_default_showgraph_dir().joinpath('GIFs', composition.name)
            # try:
            #     rmtree(composition._animate_directory)
            # except:
            #     pass
            composition._animate_unit = composition._animate.pop(UNIT, EXECUTION_SET)
            composition._image_duration = composition._animate.pop(DURATION, 0.75)
            composition._animate_num_runs = composition._animate.pop(NUM_RUNS, 1)
            composition._animate_num_trials = composition._animate.pop(NUM_TRIALS, 1)
            composition._animate_simulations = composition._animate.pop(SIMULATIONS, False)
            composition._movie_filename = composition._animate.pop(MOVIE_NAME, composition.name + ' movie') + '.gif'
            composition._animation_directory = composition._animate.pop(MOVIE_DIR, default_dir)
            composition._save_images = composition._animate.pop(SAVE_IMAGES, False)
            composition._show_animation = composition._animate.pop(SHOW, False)
            if composition._animate_unit not in {COMPONENT, EXECUTION_SET}:
                raise ShowGraphError(f"{repr(UNIT)} entry of {repr('animate')} argument for {composition.name} method "
                                       f"of {repr('run')} ({composition._animate_unit}) "
                                       f"must be {repr(COMPONENT)} or {repr(EXECUTION_SET)}.")
            if not isinstance(composition._image_duration, (int, float)):
                raise ShowGraphError(f"{repr(DURATION)} entry of {repr('animate')} argument for {repr('run')} method "
                                       f"of {composition.name} ({composition._image_duration}) must be an int or a float.")
            if not isinstance(composition._animate_num_runs, int):
                raise ShowGraphError(f"{repr(NUM_RUNS)} entry of {repr('animate')} argument for {repr('show_graph')} "
                                       f"method of {composition.name} ({composition._animate_num_runs}) must an integer.")
            if not isinstance(composition._animate_num_trials, int):
                raise ShowGraphError(f"{repr(NUM_TRIALS)} entry of {repr('animate')} argument for "
                                       f"{repr('show_graph')} method of {composition.name} ({composition._animate_num_trials}) "
                                       f"must an integer.")
            if not isinstance(composition._animate_simulations, bool):
                raise ShowGraphError(f"{repr(SIMULATIONS)} entry of {repr('animate')} argument for "
                                       f"{repr('show_graph')} method of {composition.name} ({composition._animate_num_trials}) "
                                       f"must a boolean.")
            try:
                composition._animation_directory = pathlib.Path(composition._animation_directory)
            except TypeError:
                raise ShowGraphError(
                    f"{repr(MOVIE_DIR)} entry of 'animate' argument for 'run'"
                    f" method of {composition.name} ({composition._animation_directory})"
                    " must be a string or os.PathLike."
                )
            if not isinstance(composition._movie_filename, str):
                raise ShowGraphError(f"{repr(MOVIE_NAME)} entry of {repr('animate')} argument for {repr('run')} "
                                       f"method of {composition.name} ({composition._movie_filename}) must be a string.")
            if not isinstance(composition._save_images, bool):
                raise ShowGraphError(f"{repr(SAVE_IMAGES)} entry of {repr('animate')} argument for {repr('run')}"
                                       f"method of {composition.name} ({composition._save_images}) must be a boolean")
            if not isinstance(composition._show_animation, bool):
                raise ShowGraphError(f"{repr(SHOW)} entry of {repr('animate')} argument for {repr('run')} "
                                       f"method of {composition.name} ({composition._show_animation}) must be a boolean.")

        elif composition._animate:
            # composition._animate should now be False or a dict
            raise ShowGraphError("{} argument for {} method of {} ({}) must be a boolean or "
                                   "a dictionary of argument specifications for its {} method".
                                   format(repr('animate'), repr('run'), composition.name, composition._animate, repr('show_graph')))

    def _animate_execution(self, active_items, context):

        composition = self.composition

        if composition._component_animation_execution_count is None:
            composition._component_animation_execution_count = 0
        else:
            composition._component_animation_execution_count += 1
        composition.show_graph(active_items=active_items,
                               **composition._animate,
                               output_fmt='gif',
                               context=context,
                               )

    def _generate_gifs(self, G, active_items, context):

        composition = self.composition

        def create_phase_string(phase):
            return f'%16s' % phase + ' - '

        def create_time_string(time, spec):
            if spec == 'TIME':
                r = time.run
                t = time.trial
                p = time.pass_
                ts = time.time_step
            else:
                r = t = p = ts = '__'
            return f"Time(run: %2s, " % r + f"trial: %2s, " % t + f"pass: %2s, " % p + f"time_step: %2s)" % ts

        G.format = 'gif'
        execution_phase = context.execution_phase
        time = composition.scheduler.get_clock(context).time
        run_num = time.run
        trial_num = time.trial

        if INITIAL_FRAME in active_items:
            phase_string = create_phase_string('Initializing')
            time_string = create_time_string(time, 'BLANKS')

        elif ContextFlags.PROCESSING in execution_phase:
            phase_string = create_phase_string('Processing Phase')
            time_string = create_time_string(time, 'TIME')
        # elif ContextFlags.LEARNING in execution_phase:
        #     time = composition.scheduler_learning.get_clock(context).time
        #     time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}". \
        #         format(run_num, time.trial, time.pass_, time.time_step)
        #     phase_string = 'Learning Phase - '

        elif ContextFlags.CONTROL in execution_phase:
            phase_string = create_phase_string('Control Phase')
            time_string = create_time_string(time, 'TIME')

        else:
            raise ShowGraphError(
                f"PROGRAM ERROR:  Unrecognized phase during execution of {composition.name}: {execution_phase.name}")

        label = f'\n{composition.name}\n{phase_string}{time_string}\n'
        G.attr(label=label)
        G.attr(labelloc='b')
        G.attr(fontname='Monaco')
        G.attr(fontsize='14')
        index = repr(composition._component_animation_execution_count)
        image_filename = '-'.join([repr(run_num), repr(trial_num), index])
        image_file = pathlib.Path(composition._animation_directory, image_filename + '.gif')
        G.render(filename=image_filename,
                 directory=composition._animation_directory,
                 cleanup=True,
                 # view=True
                 )
        # Append gif to composition._animation
        image = Image.open(image_file)
        # TBI?
        # if not composition._save_images:
        #     remove(image_file)
        if not hasattr(composition, '_animation'):
            composition._animation = [image]
        else:
            composition._animation.append(image)
