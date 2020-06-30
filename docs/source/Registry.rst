Registry
========

A Registry is used to maintain a list of categories (subclasses) for a given base_class of `Component`,
and to ensure that the name of every Component created in each of those categories is unique.  If an item
is created with the same name as one already in the Registry, its name is appended with a hyphenated index
(e.g., name-n) that is incremented for each new item assigned the same base name, as described in greater
detail below.

.. _Registry_Naming:

Naming
------

Every object in PsyNeuLink has a `name <Component.name>` attribute, that is a string used to refer to it in printouts
and display. The name of a object can be specified in the **name** argument of its constructor.  An object's name can
be reassigned, but this should be done with caution, as other objects may depend on its name.

*Default Names*
~~~~~~~~~~~~~~~

If the name of an object is not specified in its constructor, a default name is assigned.  Some classes of objects use
class-specific conventions for default names (see individual classes for specifics). Otherwise, the default name is
handled by the `Registry`, which assigns a default name based on the name of the class, with a hyphenated integer
suffix (<object class name>-n), beginning with '0', that is incremented for each additional object of that type
requiring a default name.  For example, the first TransferMechanism to be constructed without specifying its name
will be assigned the name 'TransferMechanism-0', the next 'TransferMechanism-1', etc..


*Duplicate Names*
~~~~~~~~~~~~~~~~~

If the name of an object specified in its constructor is the same as the name of an existing object of that type, its
name is appended with a hyphenated integer suffix (<object name>-n) that is incremented for each additional
duplicated name, beginning with '1'.  The object with the original name (implicitly instance '0') is left intact.

There is one exception to this rule, for the naming of `Port <Port>`.  Ports of the same type, but that belong to
different `Mechanisms <Mechanism>`, can have the same name (for example, TransferMechanism-0 and TransferMechanism-1
can both have an `InputPort` named input_port-0 (the default name for the first InputPort);  however, if a Port
is assigned a name that is the same as another Port of that type belonging to the *same* Mechanism, it is treated as
a duplicate, and its name is suffixed as described above.
