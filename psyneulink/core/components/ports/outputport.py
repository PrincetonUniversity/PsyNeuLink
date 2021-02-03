# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  OutputPort *****************************************************

"""

Contents
--------

* `OutputPort_Overview`
* `OutputPort_Creation`
    - `OutputPort_Deferred_Initialization`
    - `OutputPort_Primary`
    - `OutputPort_Specification`
        • `OutputPort_Forms_of_Specification`
        • `Variable and Value <OutputPort_Variable_and_Value>`
        • `Compatibility and Constraints <OutputPort_Compatibility_and_Constraints>`
        • `OutputPort_Standard`
        • `OutputPort_Customization`
            * `Custom Variable <OutputPort_Custom_Variable>`
            * `Custom Function <OutputPort_Custom_Function>`
* `OutputPort_Structure`
* `OutputPort_Execution`
* `OutputPort_Class_Reference`

.. _OutputPort_Overview:

Overview
--------

OutputPort(s) represent the result(s) of executing a Mechanism.  This may be the result(s) of its
`function <OutputPort.function>` and/or values derived from that result.  The full set of results are stored in the
Mechanism's `output_values <Mechanism_Base.output_values>` attribute.  OutputPorts are used to represent
individual items of the Mechanism's `value <Mechanism_Base.value>`, and/or useful quantities derived from
them.  For example, the `function <Mechanism_Base.function>` of a `TransferMechanism` generates
a single result (the transformed value of its input);  however, a TransferMechanism can also be assigned OutputPorts
that represent its mean, variance or other derived values.  In contrast, the `function <DDM.DDM.function>`
of a `DDM` Mechanism generates several results (such as decision accuracy and response time), each of which can be
assigned as the `value <OutputPort.value>` of a different OutputPort.  The OutputPort(s) of a Mechanism can serve
as the input to other  Mechanisms (by way of `projections <Projections>`), or as the output of a Process and/or
System.  The OutputPort's `efferents <Port.efferents>` attribute lists all of its outgoing
projections.

.. _OutputPort_Creation:

Creating an OutputPort
-----------------------

An OutputPort can be created by calling its constructor. However, in general this is not necessary, as a Mechanism
automatically creates a default OutputPort if none is explicitly specified, that contains the primary result of its
`function <Mechanism_Base.function>`.  For example, if the Mechanism is created within the `pathway` of a
`Process <Process>`, an OutputPort is created and assigned as the `sender <MappingProjection.MappingProjection.sender>`
of a `MappingProjection` to the next Mechanism in the pathway, or to the Process' `output <Process_Input_And_Output>`
if the Mechanism is a `TERMINAL` Mechanism for that Process. Other configurations can also easily be specified using
a Mechanism's **output_ports** argument (see `OutputPort_Specification` below).  If it is created using its
constructor, and a Mechanism is specified in the **owner** argument, it is automatically assigned to that Mechanism.
If its **owner* is not specified, `initialization is deferred.

.. _OutputPort_Deferred_Initialization:

*Owner Assignment and Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An OutputPort must be owned by a `Mechanism <Mechanism>`.  When OutputPort is specified in the constructor for a
`Mechanism <Mechanism>` (see `below <InputPort_Specification>`), it is automatically assigned to that Mechanism as
its owner. If the OutputPort is created directly, its `owner <Port.owner>` Mechanism can specified in the
**owner** argument of its constructor, in which case it is assigned to the specified Mechanism.  Otherwise, its
initialization is `deferred <Port_Deferred_Initialization>` until
COMMENT:
TBI: its `owner <Port_Base.owner>` attribute is assigned or
COMMENT
the OutputPort is assigned to a Mechanism using the Mechanism's `add_ports <Mechanism_Base.add_ports>` method.

.. _OutputPort_Primary:

*Primary OutputPort*
~~~~~~~~~~~~~~~~~~~~~

Every Mechanism has at least one OutputPort, referred to as its *primary OutputPort*.  If OutputPorts are not
`explicitly specified <OutputPort_Specification>` for a Mechanism, a primary OutputPort is automatically created
and assigned to its `output_port <Mechanism_Base.output_port>` attribute (note the singular),
and also to the first entry of the Mechanism's `output_ports <Mechanism_Base.output_ports>` attribute
(note the plural).  The primary OutputPort is assigned an `index <OutputPort.index>` of '0', and therefore its
`value <OutputPort.value>` is assigned as the first (and often only) item of the Mechanism's `value
<Mechanism_Base.value>` attribute.

.. _OutputPort_Specification:

*OutputPort Specification*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifying OutputPorts when a Mechanism is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OutputPorts can be specified for a `Mechanism <Mechanism>` when it is created, in the **output_ports** argument of the
Mechanism's constructor (see `examples <Port_Constructor_Argument_Examples>` in Port), or in an *OUTPUT_PORTS* entry
of a parameter dictionary assigned to the constructor's **params** argument.  The latter takes precedence over the
former (that is, if an *OUTPUT_PORTS* entry is included in the parameter dictionary, any specified in the
**output_ports** argument are ignored).

    .. _OutputPort_Replace_Default_Note:

    .. note::
        Assigning OutputPorts to a Mechanism in its constructor **replaces** any that are automatically generated for
        that Mechanism (i.e., those that it creates for itself by default).  If any of those need to be retained, they
        must be explicitly specified in the list assigned to the **output_ports** argument or the *OUTPUT_PORTS* entry
        of the parameter dictionary in the **params** argument).  In particular, if the default OutputPort -- that
        usually contains the result of the Mechanism's `function <Mechanism_Base.function>` -- is to be retained,
        it too must be specified along with any additional OutputPorts desired.


Adding OutputPorts to a Mechanism after it is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OutputPorts can also be added to a Mechanism, using the Mechanism's `add_ports <Mechanism_Base.add_methods>` method.
Unlike specification in the constructor, this **does not** replace any OutputPorts already assigned to the Mechanism.
Doing so appends them to the list of OutputPorts in the Mechanism's `output_ports <Mechanism_Base.output_ports>`
attribute, and their values are appended to its `output_values <Mechanism_Base.output_values>` attribute.  If the name
of an OutputPort added to a Mechanism is the same as one that is already exists on the Mechanism, its name is suffixed
with a numerical index (incremented for each OutputPort with that name; see `Registry_Naming`), and the OutputPort is
added to the list (that is, it does *not* replace the one that was already there).


.. _OutputPort_Variable_and_Value:

*OutputPorts* `variable <OutputPort.variable>` *and* `value <OutputPort.value>`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Each OutputPort created with or assigned to a Mechanism must reference one or more items of the Mechanism's attributes,
that serve as the OutputPort's `variable <OutputPort.variable>`, and are used by its `function <OutputPort.function>`
to generate the OutputPort's `value <OutputPort.value>`.  By default, it uses the first item of its `owner
<Port.owner>` Mechanism's `value <Mechanism_Base.value>`.  However, other attributes (or combinations of them)
can be specified in the **variable** argument of the OutputPort's constructor, or the *VARIABLE* entry in an
`OutputPort specification dictionary <OutputPort_Specification_Dictionary>` (see `OutputPort_Customization`).
The specification must be compatible (in the number and type of items it generates) with the input expected by the
OutputPort's `function <OutputPort.function>`. The OutputPort's `variable <OutputPort.variable>` is used as the
input to its `function <OutputPort.function>` to generate the OutputPort's `value <OutputPort.value>`, possibly
modulated by a `GatingSignal` (see `below <OutputPort_Modulatory_Projections>`).  The OutputPort's `value
<OutputPort.value>` must, in turn, be compatible with any Projections that are assigned to it, or `used to specify
it <OutputPort_Projection_Destination_Specification>`.

The `value <OutputPort.value>` of each OutputPort of a Mechanism is assigned to a corresponding item of the
Mechanism's `output_values <Mechanism_Base.output_values>` attribute, in the order in which they are assigned in
the **output_ports** argument of its constructor, and listed in its `output_ports <Mechanism_Base.output_ports>`
attribute.

    .. note::
       The `output_values <Mechanism_Base.output_values>` attribute of a Mechanism is **not the same** as its `value
       <Mechanism_Base.value>` attribute:
           * a Mechanism's `value <Mechanism.value>` attribute contains the full and unmodified results of its
             `function <Mechanism_Base.function>`;
           * a Mechanism's `output_values <Mechanism.output_values>` attribute contains a list of the `value
             <OutputPort.value>` of each of its OutputPorts.

.. _OutputPort_Forms_of_Specification:

Forms of Specification
^^^^^^^^^^^^^^^^^^^^^^

OutputPorts can be specified in a variety of ways, that fall into three broad categories:  specifying an OutputPort
directly; use of a `Port specification dictionary <Port_Specification>`; or by specifying one or more Components to
which it should project. Each of these is described below:

    .. _OutputPort_Direct_Specification:

    **Direct Specification of an OutputPort**

    * existing **OutputPort object** or the name of one -- it cannot belong to another Mechanism, and the format of
      its `variable <OutputPort.variable>` must be compatible with the aributes of the `owner <Port.owner>`
      Mechanism specified for the OutputPort's `variable <OutputPort.variable>` (see `OutputPort_Customization`).
    ..
    * **OutputPort class**, **keyword** *OUTPUT_PORT*, or a **string** -- creates a default OutputPort that uses
      the first item of the `owner <Port.owner>` Mechanism's `value <Mechanism_Base.value>` as its `variable
      <OutputPort.variable>`, and assigns it as the `owner <Port.owner>` Mechanism's `primary OutputPort
      <OutputPort_Primary>`. If the class name or *OUTPUT_PORT* keyword is used, a default name is assigned to
      the Port; if a string is specified, it is used as the `name <OutputPort.name>` of the OutputPort
      (see `Registry_Naming`).

    .. _OutputPort_Specification_by_Variable:

    * **variable** -- creates an OutputPort using the specification as the OutputPort's `variable
      <OutputPort.variable>` (see `OutputPort_Customization`).  This must be compatible with (have the same number
      and type of elements as) the OutputPort's `function <OutputPort.function>`.  A default name is assigned based
      on the name of the Mechanism (see `Registry_Naming`).
    ..
    .. _OutputPort_Specification_Dictionary:

    **OutputPort Specification Dictionary**

    * **OutputPort specification dictionary** -- this can be used to specify the attributes of an OutputPort,
      using any of the entries that can be included in a `Port specification dictionary <Port_Specification>`
      (see `examples <Port_Specification_Dictionary_Examples>` in Port), including:

      * *VARIABLE*:<keyword or list> - specifies the attribute(s) of its `owner <Port.owner>` Mechanism to use
        as the input to the OutputPort's `function <OutputPort.function>` (see `OutputPort_Customization`); this
        must be compatible (in the number and format of the items it specifies) with the OutputPort's `function
        <OutputPort.function>`.

      * *FUNCTION*:<`Function <Function>`, function or method> - specifies the function used to transform and/or
        combine the item(s) specified for the OutputPort's `variable <OutputPort.variable>` into its
        `value <OutputPort.value>`;  its input must be compatible (in the number and format of elements) with the
        specification of the OutputPort's `variable <OutputPort.variable>` (see `OutputPort_Customization`).

      * *PROJECTIONS* or *MECHANISMS*:<list of `Projections <Projection>` and/or `Mechanisms <Mechanism>`> - specifies
        one or more efferent `MappingProjections <MappingProjection>` from the OutputPort, Mechanims that should
        receive them, and/or `ModulatoryProjections <ModulatoryProjection>` for it to receive;  this may be constrained
        by or have consequences for the OutputPort's `variable <InputPort.variable>` and/or its `value
        <OutputPort.value>` (see `OutputPort_Compatibility_and_Constraints`).

        .. note::
           The *INDEX* and *ASSIGN* attributes described below have been deprecated in version 0.4.5, and should be
           replaced by use of the *VARIABLE* and *FUNCTION* entries, respectively.  Although use of *INDEX* and *ASSIGN*
           is currently being supported for backward compatibility, this may be eliminated in a future version.

      * *INDEX*:<int> *[DEPRECATED in version 0.4.5]* - specifies the item of the `owner <Port.owner>`
        Mechanism's `value <Mechanism_Base.value>` to be used for the OutputPort's `variable <OutputPort.variable>`;
        equivalent to specifying (OWNER_VALUE, <int>) for *VARIABLE* (see `OutputPort_Customization`), which should be
        used for compatibility with future versions.

      * *ASSIGN*:<function> *[DEPRECATED in version 0.4.5]* - specifies the OutputPort's `function`
        <OutputPort.assign>` attribute;  *FUNCTION* should be used for compatibility with future versions.

    .. _OutputPort_Projection_Destination_Specification:

    **Specifying an OutputPort by Components to which it Projects**

    COMMENT:
    `examples
      <Port_Projections_Examples>` in Port)
    COMMENT

    COMMENT:
    ------------------------------------------------------------------------------------------------------------------
    ?? PUT IN ITS OWN SECTION ABOVE OR BELOW??
    Projections from an OutputPort can be specified either as attributes, in the constructor for an OutputPort (in
    its **projections** argument or in the *PROJECTIONS* entry of an `OutputPort specification dictionary
    <OutputPort_Specification_Dictionary>`), or used to specify the OutputPort itself (using one of the
    `OutputPort_Forms_of_Specification` described above. See `Port Projections <Port_Projections>` for additional
    details concerning the specification of Projections when creating a Port.

    .. _OutputPort_Projections:

    *Projections*
    ~~~~~~~~~~~~~

    When an OutputPort is created, it can be assigned one or more `Projections <Projection>`, using either the
    **projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument with
    the key *PROJECTIONS*.  An OutputPort can be assigned either `MappingProjection(s) <MappingProjection>` or
    `GatingProjection(s) <GatingProjection>`.  MappingProjections are assigned to its `efferents <Port.efferents>`
    attribute and GatingProjections to its `mod_afferents <OutputPort.mod_afferents>` attribute.  See
    `Port Projections <Port_Projections>` for additional details concerning the specification of Projections when
    creating a Port.
    ------------------------------------------------------------------------------------------------------------------
    COMMENT

    An OutputPort can also be specified by specifying one or more Components to or from which it should be assigned
    Projection(s). Specifying an OutputPort in this way creates both the OutputPort and any of the specified or
    implied Projection(s) (if they don't already exist). `MappingProjections <MappingProjection>`
    are assigned to the OutputPort's `efferents <Port.efferents>` attribute, while `ControlProjections
    <ControlProjection>` and `GatingProjections <GatingProjection>` are assigned to its `mod_afferents
    <Port.mod_afferents>` attribute. Any of the following can be used to specify an InputPort by the Components that
    projection to it (see `below <OutputPort_Compatibility_and_Constraints>` for an explanation of the relationship
    between the `variable` of these Components and the OutputPort's `value <OutputPort.value>`):

    * **InputPort, GatingSignal, Mechanism, or list with any of these** -- creates an OutputPort with
      the relevant Projection(s).  A `MappingProjection` is created to each InputPort or ProcessingMechanism specified
      (for a Mechanism, its `primary InputPort <InputPort_Primary>` is used). A `GatingProjection` is created for
      each GatingSignal or GatingMechamism specified (for a GatingMechanism, its first GatingSignal is used).
    ..
    * **Projection** -- any form of `Projection specification <Projection_Specification>` can be used; creates an
      OutputPort and assigns it as the `sender <MappingProjection.sender>` for any MappingProjections specified, and
      as the `receiver <GatingProjection.receiver>` for any GatingProjections specified.

    .. _OutputPort_Tuple_Specification:

    * **OutputPort specification tuples** -- these are convenience formats that can be used to compactly specify an
      OutputPort along with Projections to or from it in any of the following ways:

        * **2-item tuple:** *(<Port name or list of Port names>, <Mechanism>)* -- 1st item must be the name of an
          `InputPort` or `ModulatorySignal`, or a list of such names, and the 2nd item must be the Mechanism to
          which they all belong.  Projections of the relevant types are created for each of the specified Ports
          (see `Port 2-item tuple <Port_2_Item_Tuple>` for additional details).

        * **2-item tuple:** *(<Port, Mechanism, or list of them>, <Projection specification>)* -- this is a contracted
          form of the 3-item tuple described below

        * **3-item tuple:** *(<value, Port spec, or list of Port specs>, variable spec, Projection specification)* --
          this allows the specification of Port(s) to which the OutputPort should project, together with a
          specification of its `variable <OutputPort.variable>` attribute, and (optionally) parameters of the
          Projection(s) to use (e.g., their `weight <Projection_Base.weight>` and/or `exponent
          <Projection_Base.exponent>` attributes.  Each tuple must have at least the first two items (in the
          order listed), and can include the third:

            * **value, Port specification, or list of Port specifications** -- specifies either the `variable
              <InputPort.variable>` of the InputPort, or one or more Ports that should project to it.  The Port
              specification(s) can be a (Port name, Mechanism) tuple (see above), and/or include Mechanisms, in which
              case their `primary InputPort <InputPortPrimary>` is used.  All of the Port specifications must be
              consistent with (that is, their `value <Port_Base.value>` must be compatible with the `variable
              <Projection_Base.variable>` of) the Projection specified in the fourth item if that is included.

            * **variable spec** -- specifies the attributes of the OutputPort's `owner <Port.owner>` Mechanism
              used for its `variable <OutputPort.variable>` (see `OutputPort_Customization`).

            * **Projection specification** (optional) -- `specifies a Projection <Projection_Specification>` that
              must be compatible with the Port specification(s) in the 1st item; if there is more than one
              Port specified, and the Projection specification is used, all of the Ports
              must be of the same type (i.e.,either InputPorts or GatingSignals), and the `Projection
              Specification <Projection_Specification>` cannot be an instantiated Projection (since a
              Projection cannot be assigned more than one `receiver <Projection_Base.receiver>`).

.. _OutputPort_Compatibility_and_Constraints:

OutputPort `variable <OutputPort.variable>` and `value <OutputPort.value>`: Compatibility and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The format of an OutputPorts' `variable <OutputPort.variable>` may have consequences that must be taken into
account when `specifying an OutputPort by Components to which it projects
<OutputPort_Projection_Destination_Specification>`.  These depend on the context in which the specification is
made, and possibly the value of other specifications.  These considerations and how they are handled are described
below, starting with constraints that are given the highest precedence:

  * **OutputPort specified in a Mechanism's constructor** -- the specification of the OutputPort's `variable
    <OutputPort.variable>`, together with its `function <OutputPort.function>` determine the OutputPort's `value
    <OutputPort.value>` (see `above <OutputPort_Variable_and_Value>`).  Therefore, any specifications of the
    OutputPort relevant to its `value <OutputPort.value>` must also be compatible with these factors (for example,
    `specifying it by variable <OutputPort_Specification_by_Variable>` or by a `MappingProjection` or an
    `InputPort` to which it should project (see `above <OutputPort_Projection_Destination_Specification>`).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **OutputPort specified on its own** -- any direct specification of the OutputPort's `variable
    <OutputPort.variable>` is used to determine its format (e.g., `specifying it by variable
    <OutputPort_Specification_by_Variable>`, or a *VARIABLE* entry in an `OutputPort specification dictionary
    <OutputPort_Specification_Dictionary>`.  In this case, the value of any `Components used to specify the
    OutputPort <OutputPort_Projection_Destination_Specification>` must be compatible with the specification of its
    `variable <OutputPort.variable>` and the consequences this has for its `value <OutputPort.value>` (see below).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **OutputPort's** `value <OutputPort.value>` **not constrained by any of the conditions above** -- then its
    `variable <OutputPort.variable>` is determined by the default for an OutputPort (the format of the first
    item of its `owner <Port.owner>` Mechanism's `value <Mechanism_Base.value>` ). If the OutputPort is
    `specified to project to any other Components <OutputPort_Projection_Destination_Specification>`, then if the
    Component is a:


    * **InputPort or Mechanism** (for which its `primary InputPort <InputPort_Primary>` is used) -- if its
      `variable <Port_Base.variable>` matches the format of the OutputPort's `value <OutputPort.value>`, a
      `MappingProjection` is created using an `IDENTITY_MATRIX`;  otherwise, a `FULL_CONNECTIVITY_MATRIX` is used
      that maps the OutputPort's `value <OutputPort.value>` to the InputPort's `variable <Port_Base.variable>`.

    * **MappingProjection** -- if its `matrix <MappingProjection.matrix>` is specified, then the `sender dimensionality
      <Mapping_Matrix_Dimensionality>` of the matrix must be the same as that of the OutputPort's `value
      <OutputPort.value>`; if its `receiver <MappingProjection.receiver>` is specified, but not its `matrix
      <MappingProjection.matrix>`, then a matrix is chosen that appropriately maps from the OutputPort to the
      receiver (as described just above);  if neither its `matrix <MappingProjection.matrix>` or its `receiver
      <MappingProjection.receiver>` are specified, then the Projection's `initialization is deferred
      <MappingProjection_Deferred_Initialization>` until its `receiver <MappingProjection.receiver>` is specified.

    * **GatingProjection, GatingSignal or GatingMechanism** -- any of these can be used to specify an OutputPort;
      their `value` does not need to be compatible with the OutputPort's `variable <InputPort.variable>` or
      `value <OutputPort.value>`, however it does have to be compatible with the `modulatory parameter
      <Function_Modulatory_Params>` of the OutputPort's `function <OutputPort.function>`.


.. _OutputPort_Standard:

Standard OutputPorts
^^^^^^^^^^^^^^^^^^^^^

# FIX: 11/9/19: REPLACE RECURRENTTRANSFERMECHNISM EXAMPLE WITH TRANSFERMECHANISM
Mechanisms have a `standard_output_ports <Mechanism_Base.standard_output_ports>` attribute, that contains a list of
`StandardOutputPorts`: `OutputPort specification dictionaries <OutputPort_Specification_Dictionary>` that can be
assigned as `output_ports <Mechanism_Base.output_ports>`. There is a base set of StandardOutputPorts for all
Mechanisms. Subclasses of Mechanisms may add ones that are specific to that type of Mechanism (for example, the
`RecurrentTransferMechanism` class has `standard_output_ports <RecurrentTransferMechanism.standard_output_ports>`
for calculating the energy and entropy of its `value <Mechanism_Base.value>`.  The names of the `standard_output_ports`
are listed in the Mechanism's `standard_output_port_names <Mechanism_Base.standard_output_port_names>` attribute.
These can be used to specify the `output_ports <Mechanism_Base.output_ports>` of a Mechanism, as in the following
example for a `ProcessingMechanism <ProcessingMechanism>`::

    >>> import psyneulink as pnl
    >>> my_mech = pnl.ProcessingMechanism(default_variable=[0,0],
    ...                                   function=pnl.Logistic,
    ...                                   output_ports=[pnl.RESULT,
    ...                                                 pnl.MEAN,
    ...                                                 pnl.VARIANCE])

In this example, ``my_mech`` is configured with three OutputPorts;  the first will be named *RESULT* and will
represent logistic transform of the 2-element input vector;  the second will be named  *MEAN* and will
represent mean of the result (i.e., of its two elements); and the third will be named *VARIANCE* and contain
the variance of the result.

.. _OutputPort_Customization:

*Custom OutputPorts*
~~~~~~~~~~~~~~~~~~~~

An OutputPort's `value <OutputPort.value>` can be customized by specifying its `variable <OutputPort.variable>`
and/or `function <OutputPort.function>` in the **variable** and **function** arguments of the OutputPort's
constructor, the corresponding entries (*VARIABLE* and *FUNCTION*) of an `OutputPort specification
dictionary <OutputPort_Specification_Dictionary>`, or in the variable spec (2nd) item of a `3-item tuple
<OutputPort_Tuple_Specification>` for the OutputPort.

.. _OutputPort_Custom_Variable:

*OutputPort* `variable <OutputPort.variable>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, an OutputPort uses the first (and often only) item of the owner Mechanism's `value
<Mechanism_Base.value>` as its `variable <OutputPort.variable>`.  However, this can be customized by specifying
any other item of its `owner <Port.owner>`\\s `value <Mechanism_Base.value>`, the full `value
<Mechanism_Base.value>` itself, other attributes of the `owner <Port.owner>`, or any combination of these
using the following:

    *OWNER_VALUE* -- keyword specifying the entire `value <Mechanism_Base.value>` of the OutputPort's `owner
    <Port.owner>`.

    *(OWNER_VALUE, <int>)* -- tuple specifying an item of the `owner <Port.owner>`\\'s `value
    <Mechanism_Base.value>` indexed by the int;  indexing begins with 0 (e.g.; 1 references the 2nd item).

    *<attribute name>* -- the name of an attribute of the OutputPort's `owner <Port.owner>` (must be one
    in the `owner <Port.owner>`\\'s `Parameters <Mechanism.Parameters>` class); returns the value
    of the named attribute for use in the OutputPort's `variable <OutputPort.variable>`.

    *PARAMS_DICT* -- keyword specifying the `owner <Port.owner>` Mechanism's
    entire dictionary of Parameters, that contains its own Parameters, its
    `function <Mechanism.function`\\'s Parameters, and the current `variable`
    for the Mechanism's `input_ports <Mechanism.input_ports>`. The
    OutputPort's `function <OutputPort.function>` must be able to parse the dictionary.
    COMMENT
    ??WHERE CAN THE USER GET THE LIST OF ALLOWABLE ATTRIBUTES?  USER_PARAMS?? aTTRIBUTES_DICT?? USER ACCESSIBLE PARAMS??
    <obj>.parameters
    COMMENT

    *List[<any of the above items>]* -- this assigns the value of each item in the list to the corresponding item of
    the OutputPort's `variable <OutputPort.variable>`.  This must be compatible (in number and type of elements) with
    the input expected by the OutputPort's `function <OutputPort.function>`.

.. _OutputPort_Custom_Function:

*OutputPort* `function <OutputPort.function>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the `function <OutputPort.function>` of an OutputPort is `Linear`, which simply assigns the
OutputPort's `variable <OutputPort.variable>` as its `value <OutputPort.value>`.  However, a different function
can be assigned, to transform and/or combine the item(s) of the OutputPort's `variable <OutputPort.variable>`
for use as its `value <OutputPort.value>`. The function can be a PsyNeuLink `Function <Function>` or any Python
function or method, so long as the input it expects is compatible (in number and format of elements) with the
OutputPort's `variable <OutputPort.variable>`.

Examples
--------

In the following example, a `DDM` Mechanism named ``my_mech`` is configured with three OutputPorts:

COMMENT:
(also see `OutputPort_Structure` below). If the
Mechanism's `function
<Mechanism_Base.function>` returns a value with more than one item (i.e., a list of lists, or a 2d np.array), then an
OutputPort can be assigned to any of those items by specifying its `index <OutputPort.index>` attribute. An
OutputPort can also be configured to transform the value of the item, by specifying a function for its `assign
<OutputPort.assign>` attribute; the result will then be assigned as the OutputPort's `value <OutputPort.value>`.
An OutputPort's `index <OutputPort.index>` and `assign <OutputPort.assign>` attributes can be assigned when
the OutputPort is assigned to a Mechanism, by including *INDEX* and *ASSIGN* entries in a `specification dictionary
<OutputPort_Specification>` for the OutputPort, as in the following example::
COMMENT

    >>> my_mech = pnl.DDM(function=pnl.DriftDiffusionAnalytical(),
    ...                   output_ports=[pnl.DECISION_VARIABLE,
    ...                                  pnl.PROBABILITY_UPPER_THRESHOLD,
    ...                                  {pnl.NAME: 'DECISION ENTROPY',
    ...                                   pnl.VARIABLE: (pnl.OWNER_VALUE, 2),
    ...                                   pnl.FUNCTION: pnl.Stability(metric=pnl.ENTROPY) }])

COMMENT:
   ADD VERSION IN WHICH INDEX IS SPECIFIED USING DDM_standard_output_ports
   CW 3/20/18: TODO: this example is flawed: if you try to execute() it, it gives divide by zero error.
COMMENT

The first two are `Standard OutputPorts <OutputPort_Standard>` that represent the decision variable of the DDM and
the probability of it crossing of the upper (vs. lower) threshold. The third is a custom OutputPort, that computes
the entropy of the probability of crossing the upper threshold.  It uses the 3rd item of the DDM's `value <DDM.value>`
(items are indexed starting with 0), which contains the `probability of crossing the upper threshold
<DDM_PROBABILITY_UPPER_THRESHOLD>`, and uses this as the input to the `Stability` Function assigned as the
OutputPort's `function <OutputPort.function>`, that computes the entropy of the probability. The three OutputPorts
will be assigned to the `output_ports <Mechanism_Base.output_ports>` attribute of ``my_mech``, and their values
will be assigned as items in its `output_values <Mechanism_Base.output_values>` attribute, in the order in which they
are listed in the **output_ports** argument of the constructor for ``my_mech``.

Custom OutputPorts can also be created on their own, and separately assigned or added to a Mechanism.  For example,
the ``DECISION ENTROPY`` OutputPort could be created as follows::

    >>> decision_entropy_output_port = pnl.OutputPort(name='DECISION ENTROPY',
    ...                                                 variable=(OWNER_VALUE, 2),
    ...                                                 function=pnl.Stability(metric=pnl.ENTROPY))

and then assigned either as::

    >>> my_mech = pnl.DDM(function=pnl.DriftDiffusionAnalytical(),
    ...                   output_ports=[pnl.DECISION_VARIABLE,
    ...                                  pnl.PROBABILITY_UPPER_THRESHOLD,
    ...                                  decision_entropy_output_port])

or::

    >>> another_decision_entropy_output_port = pnl.OutputPort(name='DECISION ENTROPY',
    ...                                                variable=(OWNER_VALUE, 2),
    ...                                                function=pnl.Stability(metric=pnl.ENTROPY))
    >>> my_mech2 = pnl.DDM(function=pnl.DriftDiffusionAnalytical(),
    ...                    output_ports=[pnl.DECISION_VARIABLE,
    ...                                   pnl.PROBABILITY_UPPER_THRESHOLD])

    >>> my_mech2.add_ports(another_decision_entropy_output_port) # doctest: +SKIP

Note that another new OutputPort had to be used for the second example, as trying to
add the first one created for ``my_mech`` to ``my_mech2`` would have produced an error (since a Port already
belonging to one Mechanism can't be added to another.

.. _OutputPort_Structure:

Structure
---------

Every OutputPort is owned by a `Mechanism <Mechanism>`. It can send one or more `MappingProjections
<MappingProjection>` to other Mechanisms.  If its owner is a `TERMINAL` Mechanism of a Process and/or System, then the
OutputPort will also be treated as the output of that `Process <Process_Input_And_Output>` and/or of a System.  It has
the following attributes, some of which can be specified in ways that are specific to, and that can be used to
`customize, the OutputPort <OutputPort_Customization>`:

COMMENT:
.. _OutputPort_Index:

* `index <OutputPort.index>`: this determines the item of its owner Mechanism's `value <Mechanism_Base.value>` to
  which it is assigned.  By default, this is set to 0, which assigns it to the first item of the Mechanism's `value
  <Mechanism_Base.value>`.  The `index <Mechanism_Base.index>` must be equal to or less than one minus the number of
  OutputPorts listed in the Mechanism's `output_ports <Mechanism_Base.output_ports>` attribute.  The `variable
  <OutputPort.variable>` of the OutputPort must also match (in the number and type of its elements) the item of the
  Mechanism's `value <Mechanism_Base.value>` designated by the `index <OutputPort.index>`.

.. _OutputPort_Assign:

* `assign <OutputPort.assign>`:  this specifies a function used to convert the item of the owner Mechanism's
  `value <Mechanism_Base.value>` (designated by the OutputPort's `index <OutputPort.index>` attribute), before
  providing it to the OutputPort's `function <OutputPort.function>`.  The `assign <OutputPort.assign>`
  attribute can be assigned any function that accept the OutputPort's `variable <OutputPort.variable>` as its input,
  and that generates a result that can be used the input for the OutputPort's `function <OutputPort.function>`.
  The default is an identity function (`Linear` with **slope**\\ =1 and **intercept**\\ =0), that simply assigns the
  specified item of the Mechanism's `value <Mechanism_Base.value>` unmodified as the input for OutputPort's
  `function <OutputPort.function>`.
COMMENT
..
* `variable <OutputPort.variable>` -- references attributes of the OutputPort's `owner <Port.owner>` that
  are used as the input to the OutputPort's `function <OutputPort.function>`, to determine its `value
  <OutputPort.value>`.  The specification must match (in both the number and types of elements it generates)
  the input to the OutputPort's `function <OutputPort.function>`.  By default, the first item of the `owner
  <Port.owner>` Mechanisms' `value <Mechanism_Base.value>` is used.  However, this can be customized as
  described under `OutputPort_Customization`.

* `function <OutputPort.function>` -- takes the OutputPort's `variable <OutputPort.variable>` as its input, and
  generates the OutputPort's `value <OutputPort.value>` as its result.  The default function is `Linear` that simply
  assigns the OutputPort's `variable <OutputPort.variable>` as its `value <OutputPort.value>`.  However, the
  parameters of the `function <OutputPort.function>` -- and thus the `value <OutputPort.value>` of the OutputPort --
  can be modified by `GatingProjections <GatingProjection>` received by the OutputPort (listed in its
  `mod_afferents <OutputPort.mod_afferents>` attribute.  A custom function can also be specified, so long as it can
  take as its input a value that is compatible with the OutputPort's `variable <OutputPort.variable>`.

* `projections <Port.projections>` -- all of the `Projections <Projection>` sent and received by the OutputPort;

.. _OutputPort_Efferent_Projections:

* `efferents <Port.efferents>` -- `MappingProjections <MappingProjection>` that project from the OutputPort.

.. _OutputPort_Modulatory_Projections:

* `mod_afferents <Port.mod_afferents>` -- `ControlProjections <ControlProjection>` or `GatingProjections
  <GatingProjection>` that project to the OutputPort, the `value <Projection_Base.value>` of which can modify the
  OutputPort's `value <InputPort.value>` (see the descriptions of Modulation under `ModulatorySignals
  <ModulatorySignal_Modulation>`, `ControlSignals <ControlSignal_Modulation>`, and `GatingSignals
  <GatingSignal_Modulation>` for additional details).  If the  OutputPort receives more than one ModulatoryProjection,
  their values are combined before they are used to modify the `value <OutputPort.value>` of the OutputPort.
..
* `value <OutputPort.value>`:  assigned the result of the OutputPort's `function <OutputPort.function>`, possibly
  modified by any `GatingProjections <GatingProjection>` received by the OutputPort. It is used as the input to any
  projections that the OutputStatue sends.


.. _OutputPort_Execution:

Execution
---------

An OutputPort cannot be executed directly.  It is executed when the Mechanism to which it belongs is executed.
When the Mechanism is executed, the values of its attributes specified for the OutputPort's `variable
<OutputPort.variable>` (see `OutputPort_Customization`) are used as the input to the OutputPort's `function
<OutputPort.function>`. The OutputPort is updated by calling its `function <OutputPort.function>`.  The result,
modified by any `GatingProjections <GatingProjection>` the OutputPort receives (listed in its `mod_afferents
<OutputPort.mod_afferents>` attribute), is assigned as the `value <OutputPort.value>` of the OutputPort.  This is
assigned to a corresponding item of the Mechanism's `output_values <Mechanism_Base.output_values>` attribute,
and is used as the input to any projections for which the OutputPort is the `sender <Projection_Base.sender>`.

.. _OutputPort_Class_Reference:

Class Reference
---------------


"""

import copy
import numpy as np
import typecheck as tc
import types
import warnings

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import Component, ComponentError
from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.functions.transferfunctions import CostFunctions
from psyneulink.core.components.ports.port import Port_Base, _instantiate_port_list, port_type_keywords
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ALL, ASSIGN, CALCULATE, CONTEXT, CONTROL_SIGNAL, FUNCTION, GATING_SIGNAL, INDEX, INPUT_PORT, INPUT_PORTS, \
    MAPPING_PROJECTION, MAX_ABS_INDICATOR, MAX_ABS_VAL, MAX_INDICATOR, MAX_VAL, MEAN, MECHANISM_VALUE, MEDIAN, \
    NAME, OUTPUT_PORT, OUTPUT_PORTS, OUTPUT_PORT_PARAMS, \
    OWNER_VALUE, PARAMS, PARAMS_DICT, PROB, PROJECTION, PROJECTIONS, PROJECTION_TYPE, \
    RECEIVER, REFERENCE_VALUE, RESULT, STANDARD_DEVIATION, STANDARD_OUTPUT_PORTS, PORT, VALUE, VARIABLE, VARIANCE, \
    output_port_spec_to_parameter_name, INPUT_PORT_VARIABLES
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import \
    convert_to_np_array, is_numeric, iscompatible, make_readonly_property, recursive_update, ContentAddressableList

__all__ = [
    'OutputPort', 'OutputPortError', 'PRIMARY', 'SEQUENTIAL', 'StandardOutputPorts', 'StandardOutputPortsError',
    'port_type_keywords',
]

port_type_keywords = port_type_keywords.update({OUTPUT_PORT})

# class OutputPortLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE

OUTPUT_PORT_TYPES = 'outputPortTypes'

# Used to specify how StandardOutputPorts are indexed
PRIMARY = 0
SEQUENTIAL = 'SEQUENTIAL'

DEFAULT_VARIABLE_SPEC = (OWNER_VALUE, 0)


def _parse_output_port_variable(variable, owner, context=None, output_port_name=None):
    """Return variable for OutputPort based on VARIABLE entry of owner's params dict

    The format of the VARIABLE entry determines the format returned:
    - if it is a single item, or a single item in a list, a single item is returned;
    - if it is a list with more than one item, a list is returned.
    :return:
    """

    def parse_variable_spec(spec):
        from psyneulink.core.components.mechanisms.mechanism import MechParamsDict
        if spec is None or is_numeric(spec) or isinstance(spec, MechParamsDict):
            return spec
        elif isinstance(spec, tuple):
            # Tuple indexing item of owner's attribute (e.g.,: OWNER_VALUE, int))
            try:
                owner_param_name = output_port_spec_to_parameter_name[spec[0]]
            except KeyError:
                owner_param_name = spec[0]

            try:
                index = spec[1]() if callable(spec[1]) else spec[1]

                # context is None during initialization, and we don't want to
                # incur the cost of .get during execution
                if context is None:
                    return getattr(owner.parameters, owner_param_name).get(context)[index]
                else:
                    return getattr(owner.parameters, owner_param_name)._get(context)[index]
            except TypeError:
                if context is None:
                    if getattr(owner.parameters, owner_param_name).get(context) is None:
                        return None
                elif getattr(owner.parameters, owner_param_name)._get(context) is None:
                    return None
                else:
                    # raise OutputPortError("Can't parse variable ({}) for {} of {}".
                    #                        format(spec, output_port_name or OutputPort.__name__, owner.name))
                    raise Exception
            except:
                raise OutputPortError(f"Can't parse variable ({spec}) for "
                                       f"{output_port_name or OutputPort.__name__} of {owner.name}.")

        elif isinstance(spec, str) and spec == PARAMS_DICT:
            # Specifies passing owner's params_dict as variable
            return {
                **{p.name: p._get(context) for p in owner.parameters},
                **{p.name: p._get(context) for p in owner.function.parameters},
                **{
                    INPUT_PORT_VARIABLES:
                    [
                        input_port.parameters.variable._get(context)
                        for input_port in owner.input_ports
                    ]
                }
            }
        elif isinstance(spec, str):
            # Owner's full value or attribute other than its value
            try:
                owner_param_name = output_port_spec_to_parameter_name[spec]
            except KeyError:
                owner_param_name = spec

            try:
                # context is None during initialization, and we don't want to
                # incur the cost of .get during execution
                if context is None:
                    return getattr(owner.parameters, owner_param_name).get(context)
                else:
                    return getattr(owner.parameters, owner_param_name)._get(context)
            except AttributeError:
                try:
                    if context is None:
                        return getattr(owner.function.parameters, owner_param_name).get(context)
                    else:
                        return getattr(owner.function.parameters, owner_param_name)._get(context)
                except AttributeError:
                    raise OutputPortError(f"Can't parse variable ({spec}) for "
                                           f"{output_port_name or OutputPort.__name__} of {owner.name}.")
        else:
            raise OutputPortError(f"'{VARIABLE.upper()}' entry for {output_port_name or OutputPort.__name__} "
                                   f"specification dictionary of {owner.name} ({spec}) must be "
                                   "numeric or a list of {owner.__class__.__name__} attribute names.")

    if not isinstance(variable, list):
        variable = [variable]

    if len(variable)== 1:
        return parse_variable_spec(variable[0])

    fct_variable = []
    for spec in variable:
        fct_variable.append(parse_variable_spec(spec))
    return fct_variable


def _output_port_variable_getter(owning_component=None, context=None, output_port_name=None):
    return _parse_output_port_variable(owning_component._variable_spec, owning_component.owner, context, output_port_name)


class OutputPortError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class OutputPort(Port_Base):
    """
    OutputPort(            \
        reference_value,   \
        function=Linear(), \
        projections=None)

    Subclass of `Port <Port>` that calculates and represents an output of a `Mechanism <Mechanism>`.
    See `Port_Class_Reference` for additional arguments and attributes.

    COMMENT:

    PortRegistry
    -------------
        All OutputPorts are registered in PortRegistry, which maintains an entry for the subclass,
        a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    reference_value : number, list or np.ndarray
        a template that specifies the format of the OutputPort's `variable <OutputPort.variable>`;  if it is
        specified in addition to the **variable** argument, then these must be compatible (both in the number and
        type of elements).  It is used to insure the compatibility of the source of the input for the OutputPort
        with its `function <OutputPort.function>`.

    variable : number, list or np.ndarray
        specifies the attributes of the  OutputPort's `owner <Port.owner>` Mechanism to be used by the
        OutputPort's `function <OutputPort.function>`  in generating its `value <OutputPort.value>`.

    function : Function, function, or method : default Linear
        specifies the function used to transform and/or combine the items of its `owner <Port.owner>`\'s `value
        <Mechanism_Base.value>` (designated by the OutputPort's `variable <OutputPort.variable>`) into its `value
        <OutputPort.value>`, under the possible influence of `ControlProjections <ControlProjection>` or
        `GatingProjections <GatingProjection>` received by the OutputPort.

    COMMENT:
    index : int : default PRIMARY
        specifies the item of the owner Mechanism's `value <Mechanism_Base.value>` used as input for the
        function specified by the OutputPort's `assign <OutputPort.assign>` attribute, to determine the
        OutputPort's `value <OutputPort.value>`.

    assign : Function, function, or method : default Linear
        specifies the function used to convert the designated item of the owner Mechanism's
        `value <Mechanism_Base.value>` (specified by the OutputPort's :keyword:`index` attribute),
        before it is assigned as the OutputPort's `value <OutputPort.value>`.  The function must accept a value that
        has the same format (number and type of elements) as the item of the Mechanism's
        `value <Mechanism_Base.value>`.
    COMMENT

    projections : list of Projection specifications
        specifies the `MappingProjection(s) <MappingProjection>` to be sent by the OutputPort, and/or
        `ControlProjections <ControlProjection>` and/or `GatingProjections(s) <GatingProjection>` to be received (see
        `OutputPort_Projections` for additional details); these are listed in its `efferents <Port.efferents>` and
        `mod_afferents <Port.mod_afferents>` attributes, respectively (see `OutputPort_Projections` for additional
        details).

    Attributes
    ----------

    variable : value, list or np.ndarray
        the value(s) of the item(s) of the `owner <Port.owner>` Mechanism's attributes specified in the
        **variable** argument of the constructor, or a *VARIABLE* entry in the `OutputPort specification dictionary
        <OutputPort_Specification_Dictionary>` used to construct the OutputPort.

    COMMENT:
    index : int
        the item of the owner Mechanism's `value <Mechanism_Base.value>` used as input for the function specified by
        its `assign <OutputPort.assign>` attribute (see `index <OutputPort_Index>` for additional details).

    assign : function or method
        function used to convert the item of the owner Mechanism's `value <Mechanism_Base.value>` specified by
        the OutputPort's `index <OutputPort.index>` attribute.  The result is combined with the result of the
        OutputPort's `function <OutputPort.function>` to determine both the `value <OutputPort.value>` of the
        OutputPort, as well as the value of the corresponding item of the owner Mechanism's `output_values
        <Mechanism_Base.output_values>`. The default (`Linear`) transfers the value unmodified  (see `assign
        <OutputPort_Assign>` for additional details)
    COMMENT

    function : Function, function, or method
        function used to transform and/or combine the value of the items of the OutputPort's `variable
        <OutputPort.variable>` into its `value <OutputPort.value>`, under the possible influence of
        `ControlProjections <ControlProjection>` or `GatingProjections <GatingProjection>` received by the OutputPort.

    value : number, list or np.ndarray
        assigned the result of `function <OutputPort.function>`;  the same value is assigned to the corresponding item
        of the owner Mechanism's `output_values <Mechanism_Base.output_values>` attribute.

    label : string or number
        the string label that represents the current `value <OutputPort.value>` of the OutputPort, according to the
        owner mechanism's `output_labels_dict <Mechanism.output_labels_dict>`. If the current
        `value <OutputPort.value>` of the OutputPort does not have a corresponding label, then the numeric
        `value <OutputPort.value>` is returned.

    name : str
        the name of the OutputPort; if it is not specified in the **name** argument of the constructor, a default is
        assigned by the OutputPortRegistry of the Mechanism to which the OutputPort belongs.  Note that most
        Mechanisms automatically create one or more `Standard OutputPorts <OutputPort_Standard>`, that have
        pre-specified names.  However, if any OutputPorts are specified in the **input_ports** argument of the
        Mechanism's constructor, those replace its Standard OutputPorts (see `note
        <Mechanism_Default_Port_Suppression_Note>`);  `standard naming conventions <Registry_Naming>` apply to the
        OutputPorts specified, as well as any that are added to the Mechanism once it is created (see `note
        <Port_Naming_Note>`).

    """

    #region CLASS ATTRIBUTES

    componentType = OUTPUT_PORT
    paramsType = OUTPUT_PORT_PARAMS

    # portAttributes = Port_Base.portAttributes | {INDEX, ASSIGN}

    connectsWith = [INPUT_PORT, GATING_SIGNAL, CONTROL_SIGNAL]
    connectsWithAttribute = [INPUT_PORTS]
    projectionSocket = RECEIVER
    modulators = [GATING_SIGNAL, CONTROL_SIGNAL]
    canReceive = modulators
    projection_type = MAPPING_PROJECTION

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'OutputPortCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    class Parameters(Port_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <OutputPort.variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True
        """
        variable = Parameter(np.array([0]), read_only=True, getter=_output_port_variable_getter, pnl_internal=True, constructor_argument='default_variable')

    #endregion

    @tc.typecheck
    @handle_external_context()
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 function=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 index=None,
                 assign=None,
                 **kwargs):

        context = kwargs.pop(CONTEXT, None)

        # For backward compatibility with CALCULATE, ASSIGN and INDEX
        if 'calculate' in kwargs:
            assign = kwargs['calculate']
        if params:
            _maintain_backward_compatibility(params, name, owner)

        # setting here to ensure even deferred init ports have this attribute
        self._variable_spec = variable

        # If owner or reference_value has not been assigned, defer init to Port._instantiate_projection()
        # if owner is None or reference_value is None:
        if owner is None:
            # Temporarily name OutputPort
            self._assign_deferred_init_name(name)
            # Store args for deferred initialization
            self._store_deferred_init_args(**locals())

            # Flag for deferred initialization
            self.initialization_status = ContextFlags.DEFERRED_INIT
            return

        self.reference_value = reference_value

        if variable is None:
            if reference_value is None:
                variable = DEFAULT_VARIABLE_SPEC
            else:
                variable = reference_value
        variable_getter = None
        self._variable_spec = variable

        if not is_numeric(variable):
            self._variable = variable

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.output_ports here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputPorts in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params
        super().__init__(
            owner,
            variable=variable,
            size=size,
            projections=projections,
            params=params,
            name=name,
            prefs=prefs,
            context=context,
            function=function,
            index=index,
            assign=assign,
            **kwargs
        )

    def _validate_against_reference_value(self, reference_value):
        """Validate that Port.variable is compatible with the reference_value

        reference_value is the value of the Mechanism to which the OutputPort is assigned
        """
        return

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Instantiate default variable if it is None or numeric
        :param function:
        """
        super()._instantiate_attributes_before_function(function=function, context=context)

        # If variable has not been assigned, or it is numeric (in which case it can be assumed that
        #    the value was a reference_value generated during initialization/parsing and passed in the constructor
        if self._variable_spec is None or is_numeric(self._variable_spec):
            self._variable_spec = DEFAULT_VARIABLE_SPEC

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of Port's constructor

        Assume specification in projections as ModulatoryProjection if it is a:
            ModulatoryProjection
            ModulatorySignal
            ModulatoryMechanism
        Call _instantiate_projections_to_port to assign ModulatoryProjections to .mod_afferents

        Assume all remaining specifications in projections are for outgoing MappingProjections;
            these should be either Mechanisms, Ports or MappingProjections to one of those
        Call _instantiate_projections_from_port to assign MappingProjections to .efferents

        Store result of function as self.function_value
        function_value is converted to returned value by assign function

        """
        from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import \
            _is_modulatory_spec
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection

        modulatory_projections = [proj for proj in projections if _is_modulatory_spec(proj)]
        self._instantiate_projections_to_port(projections=modulatory_projections, context=context)

        # Treat all remaining specifications in projections as ones for outgoing MappingProjections
        pathway_projections = [proj for proj in projections if proj not in modulatory_projections]
        for proj in pathway_projections:
            self._instantiate_projection_from_port(projection_spec=MappingProjection,
                                                    receiver=proj,
                                                    context=context)

    def _check_for_duplicate_projections(self, projection):
        """Check if projection is redundant with one in efferents of OutputPort

        Check for any instantiated projection in efferents with the same receiver as projection
        or one in deferred_init status with receiver specification that is the same type as projection.

        Returns redundant Projection if found, otherwise False.
        """

        # FIX: 7/22/19 - CHECK IF RECEIVER IS SPECIFIED AS MECHANISM AND, IF SO, CHECK ITS PRIMARY_INPUT_PORT
        duplicate = next(iter([proj for proj in self.efferents
                               if ((proj.receiver == projection.receiver and proj != projection)
                                   or (proj.initialization_status == ContextFlags.DEFERRED_INIT
                                       and proj._init_args[RECEIVER] == type(projection.receiver)))]), None)
        if duplicate and self.verbosePref or self.owner.verbosePref:
            from psyneulink.core.components.projections.projection import Projection
            warnings.warn(f'{Projection.__name__} from {projection.sender.name} of {projection.sender.owner.name} '
                          f'to {self.name} of {self.owner.name} already exists; will ignore additional '
                          f'one specified ({projection.name}).')
        return duplicate

    def _get_primary_port(self, mechanism):
        return mechanism.output_port

    def _parse_arg_variable(self, default_variable):
        return _parse_output_port_variable(default_variable, self.owner)

    def _parse_function_variable(self, variable, context=None):
        return _parse_output_port_variable(variable, self.owner)

    @tc.typecheck
    def _parse_port_specific_specs(self, owner, port_dict, port_specific_spec):
        """Get variable spec and/or connections specified in an OutputPort specification tuple

        Tuple specification can be:
            (port_spec, connections)
            (port_spec, variable spec, connections)

        See Port._parse_port_specific_spec for additional info.

        Returns:
             - port_spec:  1st item of tuple
             - params dict with VARIABLE and/or PROJECTIONS entries if either of them was specified

        """
        # FIX: ADD FACILITY TO SPECIFY WEIGHTS AND/OR EXPONENTS FOR INDIVIDUAL OutputPort SPECS
        #      CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
        #      THESE CAN BE USED BY THE InputPort's LinearCombination Function
        #          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputPort VALUES)
        #      THIS WOULD ALLOW FULLY GENEREAL (HIEARCHICALLY NESTED) ALGEBRAIC COMBINATION OF INPUT VALUES
        #      TO A MECHANISM
        from psyneulink.core.components.projections.projection import _parse_connection_specs, ProjectionTuple

        params_dict = {}
        port_spec = port_specific_spec

        if isinstance(port_specific_spec, dict):
            # port_dict[VARIABLE] = _parse_output_port_variable(port_dict[VARIABLE], owner)
            # # MODIFIED 3/10/18 NEW:
            # if port_dict[VARIABLE] is None:
            #     port_dict[VARIABLE] = DEFAULT_VARIABLE_SPEC
            # # MODIFIED 3/10/18 END
            return None, port_specific_spec

        elif isinstance(port_specific_spec, ProjectionTuple):
            port_spec = None
            params_dict[PROJECTIONS] = _parse_connection_specs(self,
                                                               owner=owner,
                                                               connections=[port_specific_spec])

        elif isinstance(port_specific_spec, tuple):
            tuple_spec = port_specific_spec
            port_spec = None
            TUPLE_VARIABLE_INDEX = 1

            if is_numeric(tuple_spec[0]):
                port_spec = tuple_spec[0]
                reference_value = port_dict[REFERENCE_VALUE]
                # Assign value so sender_dim is skipped below
                # (actual assignment is made in _parse_port_spec)
                if reference_value is None:
                    port_dict[REFERENCE_VALUE]=port_spec
                elif not iscompatible(port_spec, reference_value):
                    raise OutputPortError("Value in first item of 2-item tuple specification for {} of {} ({}) "
                                     "is not compatible with its {} ({})".
                                     format(OutputPort.__name__, owner.name, port_spec,
                                            REFERENCE_VALUE, reference_value))
                projection_spec = tuple_spec[1]

            else:
                projection_spec = port_specific_spec if len(port_specific_spec)==2 else (port_specific_spec[0],
                                                                                           port_specific_spec[-1])

            if not len(tuple_spec) in {2,3} :
                raise OutputPortError("Tuple provided in {} specification dictionary for {} ({}) must have "
                                       "either 2 ({} and {}) or 3 (optional additional {}) items, "
                                       "or must be a {}".
                                       format(OutputPort.__name__, owner.name, tuple_spec,
                                              PORT, PROJECTION, 'variable spec', ProjectionTuple.__name__))


            params_dict[PROJECTIONS] = _parse_connection_specs(connectee_port_type=self,
                                                               owner=owner,
                                                               connections=projection_spec)


            # Get VARIABLE specification from (port_spec, variable spec, connections) tuple:
            if len(tuple_spec) == 3:

                tuple_variable_spec = tuple_spec[TUPLE_VARIABLE_INDEX]

                # Make sure OutputPort's variable has not already been specified
                dict_variable_spec = None
                if VARIABLE in params_dict and params_dict[VARIABLE] is not None:
                    dict_variable_spec = params_dict[VARIABLE]
                elif VARIABLE in port_dict and port_dict[VARIABLE] is not None:
                    dict_variable_spec = params_dict[VARIABLE]
                if dict_variable_spec:
                    name = port_dict[NAME] or self.__name__
                    raise OutputPortError("Specification of {} in item 2 of 3-item tuple for {} ({})"
                                           "conflicts with its specification elsewhere in the constructor for {} ({})".
                                           format(VARIABLE, name, tuple_spec[TUPLE_VARIABLE_INDEX],
                                                  owner.name, dict_variable_spec))

                # Included for backward compatibility with INDEX
                if isinstance(tuple_variable_spec, int):
                    tuple_variable_spec = (OWNER_VALUE, tuple_variable_spec)

                # validate that it is a legitimate spec
                _parse_output_port_variable(tuple_variable_spec, owner)

                params_dict[VARIABLE] = tuple_variable_spec


        elif port_specific_spec is not None:
            raise OutputPortError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, port_specific_spec))

        return port_spec, params_dict

    def _execute(self, variable=None, context=None, runtime_params=None):
        value = super()._execute(
            variable=variable,
            context=context,
            runtime_params=runtime_params,
        )
        return np.atleast_1d(value)

    def _get_variable_from_projections(self, context=None):
        # fall back to specified item(s) of owner's value
        try:
            return self.parameters.variable._get(context)
        except ComponentError:
            # KDM 8/2/19: double check the relevance of this branch
            return None

    @staticmethod
    def _get_port_function_value(owner, function, variable):
        fct_variable = _parse_output_port_variable(variable, owner)

        # If variable has not been specified, assume it is the default of (OWNER_VALUE,0), and use that value
        is_PARAMS_DICT = False
        if fct_variable is None:
            try:
                if owner.defaults.value is not None:
                    fct_variable = owner.defaults.value[0]
                # Get owner's value by calling its function
                else:
                    fct_variable = owner.function(owner.defaults.variable)[0]
            except AttributeError:
                fct_variable = None
        elif type(fct_variable) is str:
            is_PARAMS_DICT = fct_variable == PARAMS_DICT

        fct = _parse_output_port_function(owner, OutputPort.__name__, function, is_PARAMS_DICT)

        try:
            # return fct(variable=fct_variable)
            return Port_Base._get_port_function_value(owner=owner, function=fct, variable=fct_variable)
        except:
            try:
                return fct(fct_variable)
            except TypeError as e:
                raise OutputPortError("Error in function assigned to {} of {}: {}".
                                       format(OutputPort.__name__, owner.name, e.args[0]))

    @property
    def variable(self):
        return _parse_output_port_variable(self._variable, self.owner)

    @variable.setter
    def variable(self, variable):
        self._variable = variable

    @property
    def socket_width(self):
        return self.defaults.value.shape[-1]

    @property
    def owner_value_index(self):
        """Return index or indices of items of owner.value for any to which OutputPort's variable has been assigned
        If the OutputPort has been assigned to:
        - the entire owner value (i.e., OWNER_VALUE on its own, not in a tuple)
            return owner.value
        - a single item of owner.value (i.e.,  owner.value==(OWNER,index))
            return the index of the item
        - more than one, return a list of indices
        - to no items of owner.value (but possibly other params), return None
        """
        # Entire owner.value
        if isinstance(self._variable_spec, str) and self._variable_spec == OWNER_VALUE:
            return self.owner.value
        elif isinstance(self._variable_spec, tuple):
            return self._variable_spec[1]
        elif isinstance(self._variable_spec, list):
            indices = [item[1] for item in self._variable_spec if isinstance(item, tuple) and OWNER_VALUE in item]
            if len(indices)==1:
                return indices[0]
            elif indices:
                return indices
        else:
            return None

    @property
    def pathway_projections(self):
        return self.efferents

    @pathway_projections.setter
    def pathway_projections(self, assignment):
        self.efferents = assignment

    # For backward compatibility with INDEX and ASSIGN
    @property
    def calculate(self):
        return self.assign

    @property
    def label(self):
        return self.get_label()

    def get_label(self, context=None):
        try:
            label_dictionary = self.owner.output_labels_dict
        except AttributeError:
            label_dictionary = {}
        return self._get_value_label(label_dictionary, self.owner.output_ports, context=context)

    @property
    def _dict_summary(self):
        return {
            **super()._dict_summary,
            **{
                'shape': str(self.defaults.value.shape),
                'dtype': str(self.defaults.value.dtype)
            }
        }

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext,
                                    extra_args=[], tags:frozenset):
        if "costs" in tags:
            assert len(extra_args) == 0
            return self._gen_llvm_costs(ctx=ctx, tags=tags)

        return super()._gen_llvm_function(ctx=ctx, extra_args=extra_args, tags=tags)

    def _gen_llvm_costs(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        args = [ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_state_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer()]

        assert "costs" in tags
        builder = ctx.create_llvm_function(args, self, str(self) + "_costs",
                                           tags=tags,
                                           return_type=ctx.float_ty)

        params, state, arg_in = builder.function.args

        # FIXME: Add support for other cost types
        assert self.cost_options == CostFunctions.INTENSITY

        func = ctx.import_llvm_function(self.intensity_cost_function)
        func_params = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                  "intensity_cost_function")
        func_state = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                 "intensity_cost_function")
        func_out = builder.alloca(func.args[3].type.pointee)
        # Port input is always struct
        func_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])

        builder.call(func, [func_params, func_state, func_in, func_out])


        # Cost function output is 1 element array
        ret_ptr = builder.gep(func_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
        ret_val = builder.load(ret_ptr)
        builder.ret(ret_val)

        return builder.function


def _instantiate_output_ports(owner, output_ports=None, context=None):
    """Call Port._instantiate_port_list() to instantiate ContentAddressableList of OutputPort(s)

    Create ContentAddressableList of OutputPort(s) specified in self.output_ports

    If output_ports is not specified:
        - use owner.output_ports as list of OutputPort specifications
        - if owner.output_ports is empty, use owner.value to create a default OutputPort

    For each OutputPort:
         check for VARIABLE and FUNCTION specifications:
             if it is a Port, get from variable and function attributes
             if it is dict, look for VARIABLE and FUNCTION entries (and INDEX and ASSIGN for backward compatibility)
             if it is anything else, assume variable spec is (OWNER_VALUE, 0) and FUNCTION is Linear
         get OutputPort's value using _parse_output_port_variable() and append to reference_value
             so that it matches specification of OutputPorts (by # and function return values)

    When completed:
        - self.output_ports contains a ContentAddressableList of one or more OutputPorts;
        - self.output_port contains first or only OutputPort in list;
        - self.output_ports contains the same ContentAddressableList (of one or more OutputPorts)
        - each OutputPort properly references, for its variable, the specified attributes of its owner Mechanism
        - if there is only one OutputPort, it is assigned the full value of its owner.

    (See Port._instantiate_port_list() for additional details)

    IMPLEMENTATION NOTE:
        default(s) for self.output_ports (self.defaults.value) are assigned here
        rather than in _validate_params, as it requires function to have been instantiated first

    Returns list of instantiated OutputPorts
    """

    # Instantiate owner's standard_output_ports as StandardOutputPorts
    #    (from list of dictionaries currently in the existing standard_output_ports attribute)
    if not isinstance(owner.standard_output_ports, StandardOutputPorts):
        owner.standard_output_ports = StandardOutputPorts(owner,
                                                           owner.standard_output_ports,
                                                           indices=PRIMARY)

    reference_value = []

    # Get owner.value
    # IMPLEMENTATION NOTE:  ?? IS THIS REDUNDANT WITH SAME TEST IN Mechanism.execute ?  JUST USE RETURN VALUE??
    owner_value = owner.defaults.value

    # IMPLEMENTATION NOTE:  THIS IS HERE BECAUSE IF return_value IS A LIST, AND THE LENGTH OF ALL OF ITS
    #                       ELEMENTS ALONG ALL DIMENSIONS ARE EQUAL (E.G., A 2X2 MATRIX PAIRED WITH AN
    #                       ARRAY OF LENGTH 2), np.array (AS WELL AS np.atleast_2d) GENERATES A ValueError
    if (isinstance(owner_value, list) and
        (all(isinstance(item, np.ndarray) for item in owner_value) and
            all(
                    all(item.shape[i]==owner_value[0].shape[0]
                        for i in range(len(item.shape)))
                    for item in owner_value))):
        pass
    else:
        converted_to_2d = convert_to_np_array(owner.defaults.value, dimension=2)
        # If owner_value is a list of heterogenous elements, use as is
        if converted_to_2d.dtype == object:
            owner_value = owner.defaults.value
        # Otherwise, use value converted to 2d np.array
        else:
            owner_value = converted_to_2d

    # This allows method to be called by Mechanism.add_input_ports() with set of user-specified output_ports,
    #    while calls from init_methods continue to use owner.output_ports (i.e., OutputPort specifications
    #    assigned in the **output_ports** argument of the Mechanism's constructor)
    output_ports = output_ports or owner.output_ports

    # Get the value of each OutputPort
    # IMPLEMENTATION NOTE:
    # Should change the default behavior such that, if len(owner_value) == len owner.output_ports
    #        (that is, there is the same number of items in owner_value as there are OutputPorts)
    #        then increment index so as to assign each item of owner_value to each OutputPort
    # IMPLEMENTATION NOTE:  SHOULD BE REFACTORED TO USE _parse_port_spec TO PARSE ouput_ports arg
    if output_ports:
        for i, output_port in enumerate(output_ports):

            # OutputPort object
            if isinstance(output_port, OutputPort):
                if output_port.initialization_status == ContextFlags.DEFERRED_INIT:
                    try:
                        output_port_value = OutputPort._get_port_function_value(owner,
                                                                                output_port._init_args[FUNCTION],
                                                                                output_port._init_args[VARIABLE])
                    # For backward compatibility with INDEX and ASSIGN
                    except AttributeError:
                        index = output_port.index
                        output_port_value = owner_value[index]
                elif output_port.defaults.value is None:
                    output_port_value = output_port.function()

                else:
                    output_port_value = output_port.defaults.value

            else:
                # parse output_port
                from psyneulink.core.components.ports.port import _parse_port_spec
                output_port = _parse_port_spec(port_type=OutputPort, owner=owner, port_spec=output_port)

                _maintain_backward_compatibility(output_port, output_port[NAME], owner)

                # If OutputPort's name matches the name entry of a dict in standard_output_ports:
                #    - use the named Standard OutputPort
                #    - merge initial specifications into std_output_port (giving precedence to user's specs)
                if output_port[NAME] and hasattr(owner, STANDARD_OUTPUT_PORTS):
                    std_output_port = copy.copy(owner.standard_output_ports.get_port_dict(output_port[NAME]))
                    if std_output_port is not None:
                        try:
                            if isinstance(std_output_port[FUNCTION], Function):
                                # we should not reuse standard_output_port Function
                                # instances across multiple ports
                                std_output_port[FUNCTION] = copy.deepcopy(std_output_port[FUNCTION], memo={'no_shared': True})
                        except KeyError:
                            pass

                        _maintain_backward_compatibility(std_output_port, output_port[NAME], owner)
                        recursive_update(output_port, std_output_port, non_destructive=True)

                if FUNCTION in output_port and output_port[FUNCTION] is not None:
                    output_port_value = OutputPort._get_port_function_value(owner,
                                                                              output_port[FUNCTION],
                                                                              output_port[VARIABLE])
                else:
                    output_port_value = _parse_output_port_variable(output_port[VARIABLE], owner)
                output_port[VALUE] = output_port_value

            output_ports[i] = output_port
            reference_value.append(output_port_value)

    else:
        reference_value = owner_value

    if hasattr(owner, OUTPUT_PORT_TYPES):
        # If owner has only one type in OutputPortTypes, generate port_types list with that for all entries
        if not isinstance(owner.outputPortTypes, list):
            port_types = owner.outputPortTypes
        else:
            # If no OutputPort specified, used first port_type in outputPortTypes as default
            if output_ports is None:
                port_types = owner.outputPortTypes[0]
            else:
                # Construct list with an entry for the port_type of each OutputPort in output_ports
                port_types = []
                for output_port in output_ports:
                    port_types.append(output_port.__class__)
    else:
        # Use OutputPort as default
        port_types = OutputPort

    port_list = _instantiate_port_list(owner=owner,
                                         port_list=output_ports,
                                         port_types=port_types,
                                         port_Param_identifier=OUTPUT_PORT,
                                         reference_value=reference_value,
                                         reference_value_name="output",
                                         context=context)

    # Call from Mechanism.add_ports, so add to rather than assign output_ports (i.e., don't replace)
    if context.source & (ContextFlags.COMMAND_LINE | ContextFlags.METHOD):
        owner.output_ports.extend(port_list)
    else:
        owner.parameters.output_ports._set(port_list, context)

    # Assign value of require_projection_in_composition
    for port in owner.output_ports:
        # Assign True for owner's primary OutputPort and the value has not already been set in OutputPort constructor
        if port.require_projection_in_composition is None and owner.output_port == port:
            port.parameters.require_projection_in_composition._set(True, context)

    return port_list


class StandardOutputPortsError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class StandardOutputPorts():
    """Collection of OutputPort specification dicts for `standard OutputPorts <OutputPort_Standard>` of a class.

    Parses specification of VARIABLE, assigning indices to OWNER_VALUE if specified.
    Adds <NAME_INDEX> of each OutputPort as property of the owner's class, that returns the index of the OutputPort
    in the list.


    Arguments
    ---------
    owner : Component
        the Component to which this OutputPort belongs

    output_port_dicts : list of dicts
        list of dictionaries specifying OutputPorts for the Component specified by `owner`

    indices : PRIMARY, SEQUENTIAL, list of ints
        specifies how to assign the (OWNER_VALUE, int) entry for each dict listed in `output_port_dicts`;

        The effects of each value of indices are as follows:

            * *PRIMARY* -- assigns (OWNER_VALUE, PRIMARY) to all output_ports for which a VARIABLE entry is not
              specified;

            * *SEQUENTIAL* -- assigns sequentially incremented int to (OWNER_VALUE, int) spec of each OutputPort,
              ignoring any VARIABLE entries previously specified for individual OutputPorts;

            * list of ints -- assigns each int to an (OWNER_VALUE, int) entry of the corresponding OutputPort in
              `output_port_dicts, ignoring any VARIABLE entries previously specified for individual OutputPorts;

            * None -- assigns `None` to VARIABLE entries for all OutputPorts for which it is not already specified.

    Attributes
    ----------
    data : dict
        dictionary of OutputPort specification dictionaries

    indices : list
        list of default indices for each OutputPort specified

    names : list
        list of the default names for each OutputPort

    Methods
    -------
    get_port_dict(name)
        returns a copy of the designated OutputPort specification dictionary
    """

    keywords = {PRIMARY, SEQUENTIAL, ALL}

    @tc.typecheck
    def __init__(self,
                 owner:Component,
                 output_port_dicts:list,
                 indices:tc.optional(tc.any(int, str, list))=None):
        self.owner = owner
        self.data = self._instantiate_std_port_list(output_port_dicts, indices)

    def _instantiate_std_port_list(self, output_port_dicts, indices):

        dict_list = output_port_dicts.copy()

        # Validate that all items in output_port_dicts are dicts
        for item in output_port_dicts:
            if not isinstance(item, dict):
                raise StandardOutputPortsError(
                    "All items of {} for {} must be dicts (but {} is not)".
                    format(self.__class__.__name__, self.owner.componentName, item))

        # Assign indices

        # List was provided, so check that:
        # - it has the appropriate number of items
        # - they are all ints
        # and then assign each int to an (OWNER_VALUE, int) VARIABLE entry in the corresponding dict
        # in output_port_dicts
        # OutputPort
        if isinstance(indices, list):
            if len(indices) != len(output_port_dicts):
                raise StandardOutputPortsError("Length of the list of indices "
                                                "provided to {} for {} ({}) "
                                                "must equal the number of "
                                                "OutputPorts dicts provided "
                                                "({}) length".format(
                        self.__class__.__name__,
                        self.owner.name,
                        len(indices),
                        len(output_port_dicts)))

            if not all(isinstance(item, int) for item in indices):
                raise StandardOutputPortsError("All the items in the list of "
                                                "indices provided to {} for {} "
                                                "of {}) must be ints".
                                                format(self.__class__.__name__,
                                                       self.name,
                                                       self.owner.name))

            for index, port_dict in zip(indices, dict_list):
                port_dict.update({VARIABLE:(OWNER_VALUE, index)})

        # Assign indices sequentially based on order of items in output_port_dicts arg
        elif indices == SEQUENTIAL:
            for index, port_dict in enumerate(dict_list):
                port_dict.update({VARIABLE:(OWNER_VALUE, index)})

        # Assign (OWNER_VALUE, PRIMARY) as VARIABLE for all OutputPorts in output_port_dicts that don't
        #    have VARIABLE (or INDEX) specified (INDEX is included here for backward compatibility)
        elif indices == PRIMARY:
            for port_dict in dict_list:
                if INDEX in port_dict or VARIABLE in port_dict:
                    continue
                port_dict.update({VARIABLE:(OWNER_VALUE, PRIMARY)})

        # Validate all INDEX specifications, parse any assigned as ALL, and
        # Add names of each OutputPort as property of the owner's class that returns its name string
        for port in dict_list:
            if INDEX in port:
                if port[INDEX] in ALL:
                    port._update(params={VARIABLE:OWNER_VALUE})
                elif port[INDEX] in PRIMARY:
                    port_dict.update({VARIABLE:(OWNER_VALUE, PRIMARY)})
                elif port[INDEX] in SEQUENTIAL:
                    raise OutputPortError("\'{}\' incorrectly assigned to individual {} in {} of {}.".
                                           format(SEQUENTIAL.upper(), OutputPort.__name__, OUTPUT_PORT.upper(),
                                                  self.name))
                del port[INDEX]
            setattr(self.owner.__class__, port[NAME], make_readonly_property(port[NAME]))

        # For each OutputPort dict with a VARIABLE entry that references it's owner's value (by index)
        # add <NAME_INDEX> as property of the OutputPort owner's class that returns its index.
        for port in dict_list:
            if isinstance(port[VARIABLE], tuple):
                index = port[VARIABLE][1]
            elif isinstance(port[VARIABLE], int):
                index = port[VARIABLE]
            else:
                continue
            setattr(self.owner.__class__, port[NAME] + '_INDEX',
                    make_readonly_property(index, name=port[NAME] + '_INDEX'))

        return dict_list

    @tc.typecheck
    def add_port_dicts(self, output_port_dicts:list, indices:tc.optional(tc.any(int, str, list))=None):
        self.data.extend(self._instantiate_std_port_list(output_port_dicts, indices))
        assert True

    @tc.typecheck
    def get_port_dict(self, name:str):
        """Return a copy of the named OutputPort dict
        """
        if next((item for item in self.names if name is item), None):
            # assign dict to owner's output_port list
            return self.data[self.names.index(name)].copy()
        # raise StandardOutputPortsError("{} not recognized as name of {} for {}".
        #                                 format(name, StandardOutputPorts.__class__.__name__, self.owner.name))
        return None

    # @tc.typecheck
    # def get_dict(self, name:str):
    #     return self.data[self.names.index(name)].copy()
    #
    @property
    def names(self):
        return [item[NAME] for item in self.data]

    # @property
    # def indices(self):
    #     return [item[INDEX] for item in self.data]

def _parse_output_port_function(owner, output_port_name, function, params_dict_as_variable=False):
    """Parse specification of function as Function, Function class, Function.function, types.FunctionType or types.MethodType.

    If params_dict_as_variable is True, and function is a Function, check whether it allows params_dict as variable;
    if it is and does, leave as is,
    otherwise, wrap in lambda function that provides first item of OutputPort's value as the functions argument.
    """
    if function is None:
        function = OutputPort.defaults.function

    if isinstance(function, (types.FunctionType, types.MethodType)):
        return function

    if isinstance(function, type) and issubclass(function, Function):
        function = function()

    if not isinstance(function, Function):
        raise OutputPortError("Specification of \'{}\' for {} of {} must be a {}, the class or function of one "
                               "or a callable object (Python function or method)".
                               format(FUNCTION.upper(), output_port_name, owner.name, Function.__name__))
    if params_dict_as_variable:
        # Function can accept params_dict as its variable
        if hasattr(function, 'params_dict_as_variable'):
            return function
        # Allow params_dict to be passed to any function, that will use the first item of the owner's value by default
        else:
            if owner.verbosePref is True:
                warnings.warn("{} specified as {} is incompatible with {} specified as {} for {} of {}; "
                              "1st item of {}'s {} attribute will be used instead".
                              format(PARAMS_DICT.upper(), VARIABLE.upper(), function.name, FUNCTION.upper(),
                                     OutputPort.name, owner.name, owner.name, VALUE))
            return lambda x: function(x[OWNER_VALUE][0])
    return function

@tc.typecheck
def _maintain_backward_compatibility(d:dict, name, owner):
    """Maintain compatibility with use of INDEX, ASSIGN and CALCULATE in OutputPort specification"""

    def replace_entries(x):

        index_present = False
        assign_present = False
        calculate_present = False

        if INDEX in x:
            index_present = True
            # if output_port[INDEX] is SEQUENTIAL:
            #     return
            if x[INDEX] == ALL:
                x[VARIABLE] = OWNER_VALUE
            else:
                x[VARIABLE] = (OWNER_VALUE, x[INDEX])
            del x[INDEX]
        if ASSIGN in x:
            assign_present = True
            x[FUNCTION] = x[ASSIGN]
            del x[ASSIGN]
        if CALCULATE in x:
            calculate_present = True
            x[FUNCTION] = x[CALCULATE]
            del x[CALCULATE]
        return x, index_present, assign_present, calculate_present

    d, i, a, c = replace_entries(d)

    if PARAMS in d and isinstance(d[PARAMS], dict):
        p, i, a, c = replace_entries(d[PARAMS])
        recursive_update(d, p, non_destructive=True)
        for spec in {VARIABLE, FUNCTION}:
            if spec in d[PARAMS]:
                del d[PARAMS][spec]

    if i:
        warnings.warn("The use of \'INDEX\' has been deprecated; it is still supported, but entry in {} specification "
                      "dictionary for {} of {} should be changed to \'VARIABLE: (OWNER_VALUE, <index int>)\' "
                      " for future compatibility.".
                      format(OutputPort.__name__, name, owner.name))
        assert False
    if a:
        warnings.warn("The use of \'ASSIGN\' has been deprecated; it is still supported, but entry in {} specification "
                      "dictionary for {} of {} should be changed to \'FUNCTION\' for future compatibility.".
                      format(OutputPort.__name__, name, owner.name))
        assert False
    if c:
        warnings.warn("The use of \'CALCULATE\' has been deprecated; it is still supported, but entry in {} "
                      "specification dictionary for {} of {} should be changed to \'FUNCTION\' "
                      "for future compatibility.".format(OutputPort.__name__, name, owner.name))

    if name == MECHANISM_VALUE:
        warnings.warn("The name of the \'MECHANISM_VALUE\' StandardOutputPort has been changed to \'OWNER_VALUE\';  "
                      "it will still work, but should be changed in {} specification of {} for future compatibility.".
                      format(OUTPUT_PORTS, owner.name))
        assert False
