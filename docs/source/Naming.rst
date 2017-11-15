.. PsyNeuLink documentation master file, created by
   sphinx-quickstart on Wed Oct 19 11:51:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Naming
======

Every object in PsyNeuLink has a `name` attribute, that is a string used to refer to it in printouts and display.
The name of a object can be specified in the **name** argument of its constructor.

Default Names
-------------

If the name of an object is not specified in its construtor, a default name is assigned.  Some classes of objects use
class-specific conventions for default names (see individual classes for specifics). Otherwise, the default name is
handled by the `Registry` for the class, which assigns a default name based on the name of the class, with a
hyphenated integer suffix (<object class name>-n that is incremented for each additional object of that type
requiring a default name, beginning with '0'.  For example, the first TransferMechanism to be constructed without
specifying its name will be assigned the name 'TransferMechanism-0', the next 'TransferMechanism-1', etc.. An
object's name cannot be changed once the object has been constructed.

# *** CONFIRM -0 FOR FIRST INSTANCE AND, IF NOT, FIX


Duplicate Names
---------------

If the name of an object specified in its construtor is the same as the name of an existing object of that type, its
name is appended with a hyphenated integer suffix (<object name>-n) that is incremented for each additional
duplicated name, beginning with '1'.  The object with the original name (implicitly instance '0') is left intact.

# *** CONFIRM -1 FOR DUPLICATE NAMES
# *** CAN OBJECXTS OF DIFFERENT TYPES HVAE THE SAME NAME?
# *** NOTE ABOUT SCOPE OF STATES HERE