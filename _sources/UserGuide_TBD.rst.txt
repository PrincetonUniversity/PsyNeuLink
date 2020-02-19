User's Guide
============

* :ref:`User_Guide_Organization_and_Principles`
* :ref:`User_Guide_Components`
* :ref:`User_Guide_Compositions`
* :ref:`User_Guide_Processing`
* :ref:`User_Guid_Modulation`


.. _User_Guide_Organization_and_Principles:

Organization and Basic Principles
---------------------------------

This section provides a overview of the basic objects in PsyNeuLink, and how they are used to construct and run models.
There are two primary kinds of objects in PsyNeuLink:  `Components <Component>` and `Compositions <Composition>`.
Components are objects that perform a specific function, and Compositions are objects
that combine Components into a runnable model.

Components
~~~~~~~~~~

    There are two primary types of Components:  `Mechanisms <Mechanism>` and `Projections <Projection>`.
    For those familiar with block modeling systems, Mechanisms are the "blocks" in PsyNeuLink, and Projections are the
    "links".  Mechanisms take inputs, process them in some way, and generate outputs that can be sent to other
    Mechanisms. Projections are the means of sending information from one Mechanism to another.  There are several
    varieties of Mechanisms and Projections, that serve a range of different functions.  At the highest level, they
    are divided into two types:  ones responsible for processing, and ones responsible for modulation.

    **Processing**
    The Components responsible for processing are `ProcessingMechanisms <ProcessingMechanism>` and
    `PathwayProjection <PathwayProjections>`.  They are used to create pathways that transmit and transform
    information, taking the inputs to a model and generating its outputs. The primary types of ProcessingMechanisms
    are `TransferMechanisms <TransferMechanism>` (that perform a "memory-less" transformation of
    their input), IntegratorMechanisms (that maintain a memory of their prior state, and integrate that with new
    inputs), and ObjectiveMechanisms (that evaluate and/or compare different sources of input).







, as well as two other fundamental types of Components
(`Ports <Port>` and `Functions <Function>`), that are described in the section below on
`Components <User_Guide_Components>`.  The other primary type of object, `Composition`, has two primary types:
`Processes <Process>` and `Systems <System>` that allow Compositions of different degrees of size and complexity to
be created.  These are described in the section below on `Compositions <User_Guide_Components>`.  In each case,
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
                    ports (mechanisms and projections)
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
