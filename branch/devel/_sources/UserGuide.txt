
PsyNeuLink User's Guide
=======================

* :ref:`User_Guide_Organization_and_Principles`
* :ref:`User_Guide_Components`
* :ref:`User_Guide_Compositions`
* :ref:`User_Guide_Processing`
* :ref:`User_Guid_Modulation`


.. _User_Guide_Organization_and_Principles:

Organization and Basic Principles
---------------------------------

This section provides a overview of the basic kinds of objects in PsyNeuLink, and how they can be used to construct
models of mind/brain function.  There are two primary kinds of objects in PsyNeuLink:  `Components <Component>` and
`Compositions <Composition>`.  Components are objects that perform a specific function, and Compositions are objects
that combine Components to form a model that can be run to examine how the Components interact, and the kinds of
behavior to which they give rise.  There are two primary types of Components:  `Mechanisms <Mechanism>`, that
process their input to generate an output that is transformed in some way (these can be thought of as the "blocks" in
a block modeling system);  and `Projections <Projection>` that take the output of one Mechanism and convey it as the
input to another (akin to the "links" in a block modeling system).  There are several subtyptes of each of these
primary components that serve an array of specific purposes, as well as two other fundamental types of Components
(`States <State>` and `Functions <Function>`), that are described in the section below on
`Components <User_Guide_Components>`.  The other primary type of object, `Composition`, has two primary types:
`Processes <Process>` and `Systems <System>` that allow Compositions of different degrees of size and complexity to
be created.  These are described in the section below on `Compositions <User_Guide_Compnents>`.  In each case,
examples are provided that illustrate how these objects are implemented, and that parallel those used in the
interactive `Tutorial <LINK>`.





Organization:
    Two main types:
        Components
            objects that actually do computation;
                two main types, of which there two types of each:
                    mechanisms:
                        processing
                        adaptive
                    projections:
                        pathway
                        modulatory
                two constituent types (i.e. parts of other types)
                    states (mechanisms and projections)
                    functions (all PNL objects)
       Compositions:
          configure components into functioning processes and systems

Basic principles:
   Composition
   Processing
   Modulation
      Learning (most common but most complicated)
      Control
      Gating

 - list item 1
 ..
 - list item 2
 ..

.. _User_Guide_Components:

Components
----------

Other packages that are much better for such applications are:
`Text description <https://URL>`_


.. _User_Guide_Compositions:

Compositions
------------

.. _User_Guide_Processing:

Processing
----------

.. _User_Guide_Modulation:

Modulation
----------


..
   .. toctree::
      :maxdepth: 1

      System
      Process

   .. toctree::
      :maxdepth: 3

      Mechanism

   .. toctree::
      :maxdepth: 2

      State
      Projection
      Functions
      Run

.. toctree::
   :hidden:

   System
   Process
   Mechanism
   State
   Projection
   Run
   Component
   Function
