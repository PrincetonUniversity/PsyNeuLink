Visualization
=============

There are two ways to visualize models composed in PsyNeuLink: statically, using the `show_graph <ShowGraph.show_graph>`
method of a `Composition`, and using `PsyNeuLinkView <https://github.com/dillontsmith/PsyNeuLinkView>`_. The latter is
currently a standalone application that interacts closely with the python script in which a PsyNeuLink model is
composed.  It can be accessed `here <https://github.com/dillontsmith/PsyNeuLinkView>`_, but is still under development.
At present, the primary way to visualize PsyNeuLink models is using the `show_graph <ShowGraph .show_graph>` method.
See `Visualizing a Composition <Composition_Visualization>` for details and examples, and the
`ShowGraph_Class_Reference` below for the arguments that can be passed to `show_graph <ShowGraph.show_graph>` to
customize the display.


.. automodule:: psyneulink.core.compositions.showgraph
   :members:
   :exclude-members: random, ShowGraphError
