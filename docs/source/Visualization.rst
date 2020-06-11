Visualization
=============

There are three ways to visualize models composed in PsyNeuLink: statically, using the `show_graph
<ShowGraph.show_graph>` method of a `Composition`; using the **animate** argument of a Composition's `run
<Composition.run>` method to output a gif showing the sequence with which its `Nodes <Composition_Nodes>`
are executed (see `example <BasicsAndPrimer_Stroop_Example_Animation_Figure>`); or interactively to configure
the display and plot Component `values <Component.value>` using
`PsyNeuLinkView <http://www.psyneuln.deptcpanel.princeton.edu/psyneulink-view-2/>`_ --
a standalone application that interacts closely with the Python script in which a PsyNeuLink model is composed.

.. note::
   The `PsyNeuLinkView <http://www.psyneuln.deptcpanel.princeton.edu/psyneulink-view-2/>`_ application is still under
   development;  at present, it can be used to display a Composition and arrange its Components.  Its functionality is
   being actively expanded, and it should soon be able to display animated plots of Component values as a Composition
   executes.  It can be accessed `here <https://github .com/dillontsmith/PsyNeuLinkView>`_.

At present, use of the Composition's `show_graph <ShowGraph.show_graph>` method and the **animate** argument of its
`run <Composition.run>` method are the primary ways to visualize a `Composition`.  The former is described below,
including `examples <ShowGraph_Examples_Visualization>` of its use.

.. automodule:: psyneulink.core.compositions.showgraph
   :members:
   :exclude-members: random, ShowGraphError
