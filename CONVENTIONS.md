
# PsyNeuLink Organization, Coding, and Documentation Conventions

## REPOSITORY ORGANIZATION:

### Core:
Made up of two types of classes:
- *abstract base classes* (italicized) - cannot be instantiated.
- **core classes** (bold) - most basic (abstract) level of objects that can be instantiated. 

#### Components
"Building blocks"
- *Mechanism*
    - *ProcessingMechanism*
        - **TransferMechanism**
        - **IntegratorMechanism**
        - **ObjectiveMechanism**
    - *AdaptiveMechanism*
        - **LearningMechanism**
        - **ControlMechanism**
        - **GatingMechanism**
- *Projection*
    - *PathwayProjection*
        - **MappingProjection**
    - *ModulatoryProjection*
        - **LearningProjection**
        - **ControlProjection**
        - **GatingProjection**
- *Port*
    - **InputPort**
    - **ParameterPort**
    - **OutputPort**
    - *ModulatorySignal*
        - **LearningSignal**
        - **ControlSignal**
        - **GatingSignal**
- *Function*
    - *TransferFunction*
    - *CombinationFunction*
    - *IntegratorFunction*
    - *DistributionFunction*
    - *LearningFunction*
        
#### Composisitons
Objects that compose building blocks and control their execution.
- *Composition*
    - **System**
    - **Process**

#### Scheduler
Objects used by Compositions to control the execution of Components and Compositions.
- **Scheduler**
- **Condition** 

### Library
Extensions of Core objects
- *Components:* classes derived from Core objects
- *Compositions:*  models
- *Models:*  published, implemented models

### NAMING:

#### Content:
- All class (and corresponding module) names should be singular  
- Component names always end in their type (e.g., TransferMechanism, LearningProjection)
  (the only exception is the DDM)
- Components and Compositions should *always* be referred to in caps
  (e.g., All Mechanisms have Projections; the receiver for a Projection is an InputPort; etc.).

#### Format:
- class names:
    fully capitalized camelCase [ClassName]
- classAttributes: camelCase without initial capitalization
- arguments_of_constructors, instance_attributes and instance_methods:
      lowercase and underscore separator(s) [constructor_arg, method_arg, object_attribute]
- keywords:
    all capitals and underscore separator(s) [KEY_WORD]
DEPRECATED:
    - internal keywords:
        prepend kw followed by camelCase [kwKeyword]

####Errors and Warnings:

warnings.warn("WARNING:..."):
  - user's input/action may produce an unexpected outcome/behavior
  
raise \<Class\>Error("PROGRAM ERROR:..."):  
  - disallowed coding practice or PsyNeuLink problem
  
Assertion:  
  - as yet unresolved/unhandled condition;  can be in devel but NOT in master (except in test scripts)


### GRAMMATICAL:

#### Elements and items of lists and arrays:
- "value": any specified token (numeric or string);
    generally references to the entity received, represented or output by a state or projection,
    but can also refer to the specification of an attribute
- "element": refers to the finest grade constituent (highest dimension / axis)
- "item" refers to any constituent at any level higher than the highest dimension / axis
- Example:  [[a, b, c] [d, e, f]]
            a, b, and c are elements of the first item
            d, e, and f are elements of the second item

#### Parameters, arguments and attributes:
- "parameter" refers to any specifiable attribute of a PsyNeuLink component
- "argument" refers to a specifiable value in a method or function call
- "attribute" is the generic Python term for an object member
- arguments "specify" a value or an assignment;  attributes "determine" a value or some outcome

#### Referencing:
- <definite article> `item`;  <indefinite article> item; e.g.: the `errorSource`;  an errorSource
- a value is "assigned" to an attribute; the value of an attribute is specified...
- a run is multiple executions;  accordingly, plural for "input" and "target" refers to multiple executions,
    not number of items in the array  (e.g., the input for an execution, the inputs for a run)

### DOCSTRING ORGANIZATION:

#### Module docstring sections:

  .. _<X>_Overview:

  Overview 
  --------
  High level description of object and its relationship to others (including its super).
  
  .. _<X>_Creation:
  
  Creating a(n) <X>
  -----------------
  Description of use of constructor, context of creation, and/or conditions of automatic creation by other objects
  
  .. _<X>_Structure:

  Structure
  ---------
  Explanation of all class-specific attributes; this should be the most elaborate explanation of each attribute, 
  that is referenced by briefer descriptions of each in the Attributes section.
       
  .. _<X>_Execution:
  
  Execution
  ---------
  Details of how the object executes, including what its `function` does
  
  .. _<X>_Class_Reference:
  
  Class Reference
  ---------------

  
#### Module / Class docstrings:
  [COMMENTED OUT:
      Technical information, with headings:
        - Description: technical description (including categor/type)
        - Class Attributes: full list of any class-specific attributes
        - Class Methods:  full list of any class-specific methods
        - Registry]\
    
  Arguments
  ---------

   for constructor (appear as "Parameters" in html docs):
    - same order as appear in constructor
    - last ones are always (in order): params, name, prefs  (with boilerplate descriptions)
    - brief description, with pointers to corresponding attribute for more detailed information.
  
  Attributes
  ----------
  - first line of each is: "attribute name : type : default <value>""; all subsequent lines should be indented;
  - include all externally-accessible attributes; 
  - organize as close as possible to order of specification in constructor, instantiation, and/or execution;
  - each should include a brief description of use and constraints on values, with reference to relevant subsection 
    of Structure section for more complete description of class-specific attributes, or to the relevant super class of 
    those attributes;  
  - last two are always (in order) name and prefs (with boilerplate descriptions).

- Commenting:
    - Sections of the docstring can be commented out by preceding the section with "COMMENT:" and ending it with 
    "COMMENT" (note the terminal colon in the first identifier and its absence in the second).
    Material that is either technical in nature, features that are planned but not implemented, and features that are
     implemented but either still in development or are not being supported should be commented out.

### MODULE ORGANIZATION:
- License
- Module docstring
- Imports
- Constants & Structures
- Keywords
- Module Error Class
- Factory method (if applicable)
- Main class definition
    - standard methods (in order, as applicable):
    - \_\_init_\_
    - _validate_variable
    - _validate_params
    - _instantiate_attributes_before_function
    - _instantiate_function
    - _execute
    - function
    - _instantiate_attributes_after_function
- Functions

### rST / SPHINX:
Terminology used here:
    - references:  a formatted string (but not necessarily with a link); the four main forms are:
        - `attribute` or `method` (shows up inside a small box)
        - `text <referenced_location>`
        - KEYWORD (always all caps with underscores)
        - **argument** (of a functon or method)
    - link: a "live" reference (i.e., when clicked, navigates somewhere); 
            can be either a keyword or a text reference 
- PsyNeuLink terms should generally be references (i.e., by enclosing in back-ticks (`term`);
    - a reference should be a *link* at least the first it is used in any paragraph;
    - in subsequent appearance in the same paragraph, the term should still be formated as a reference, 
       but generally the link should be suppressed (by using the keyword role:  :keyword:`term`).
- The format for terms should be kept as simple as possible while remaining unambiguous:
    - wheverever possible, use simple backticks (e.g., `term`);
    - if the term is ambiguous (i.e., it is used by more than one module, 
        such as the attributes 'variable', 'function', or value'),
        then add further specification: `term <Module.term>`.
    - to force a term that will be automatically parsed by Spinx as an attribute or argument, 
        to appear as normal text, use the ref role:  :ref:`term`.
    - note: arguments of methods and functions can not be linked (in the way that attributes can);
        they should be boldfaced and must be verbally designated (e.g.: the **params** argument of a function...)
- Section references should be formatted as links, and also kept as simple as possible:
    - wherever possible, simply enclose in backticks (e.g., `section`)
    - to assign a link to some other description, add angle brackets (e.g., :ref:`my text <section>`) 
    - for classes that have subclasses, the titles in the rst file (that will appear in the text of the reference) 
      are plural even though the name of the module/file itself (to which the reference must be made) is singular;
        therefore, to have the singular form appear in the text (e.g., ControlMechanism),
        the module must be explicitly referenced (e.g., `ControlMechanism <ControlMechanism>`);
        [this appears to be redundant, but it is necessary]
    - conversely, for classes without subclasses, the title in the rst file is singular;
        therefore, to refer to the plural of such a class (e.g., InputPort),
        the module must be explicitly referenced (e.g., `InputStates <InputPort>`);
    - to flag references to sections that have not yet been documented (or labelled), 
        use the following construction: `section <LINK>` (so that <LINK> can be searched for replace these later).
 
        
## Figures

***Class*** (capitalized, bold, italic)

**attribute** (lowercase, bold)

*value* (lowercase, italics)