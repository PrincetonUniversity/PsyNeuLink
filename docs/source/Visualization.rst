Visualization
=============

There are two ways to visualize models composed in PsyNeuLink: statically, using the `show_graph <ShowGraph.show_graph>`
method of a `Composition`, and using `PsyNeuLinkView <https://github.com/dillontsmith/PsyNeuLinkView>`_.

`PsyNeuLinkView <https://github.com/dillontsmith/PsyNeuLinkView>`_ is a standalone application that interacts closely
with the python script in which a PsyNeuLink model is composed.  It can be accessed `here <https://github
.com/dillontsmith/PsyNeuLinkView>`_, but is still under development.

At present, use of the Composition's `show_graph <ShowGraph.show_graph>` method is the primary way to visualize a
`Composition`.  This is described below, including `examples <ShowGraph_Examples_Visualization>` of its use.

.. automodule:: psyneulink.core.compositions.showgraph
   :members:
   :exclude-members: random, ShowGraphError
