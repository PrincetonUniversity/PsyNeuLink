# PsyNeuLink Sytle Conventions

### NAMING:

#### Content:
- Component names always end in their type (e.g., TransferMechanism, LearningProjection)
  (the only exception is the DDM)

#### Format:
- class names:
    fully capitalized camelCase [ClassName]
- externally available attributes that do not correspond to an argument of a constructor:
    camelCase without initial capitalization [externalAttribute]
- arguments of constructors and methods, and any attributes corresponding to them (must be same name):
    lowercase and underscore separator(s) [constructor_arg, method_arg]
- externally available methods and functions:
    lowercase and underscore separator(s) (enforced by PEP8) [external_method, any_function]
- internal attributes, methods and functions:
    initial undercore, lowercase, and underscore separatos(s) [_internal_method, _internal_attribute]
- externally accessible keywords:
    all capitals and underscore separator(s) [KEY_WORD]
- internal keywords:
    prepend kw followed by camelCase [kwKeyword]

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
- Overview
- Creating an <X>
- Structure
     - subsections for major subcomponents
- Execution

#### Class docstring sections:
- User-friendly list of constructor arguments with default values
- One line summary description.
- Several sentence / short paragraph description/overview.
- Technical information (COMMENTED OUT), with headings:
    - Description: technical description (including categor/type)
    - Class Attributes: full list of any class-specific attributes
    - Class Methods:  full list of any class-specific methods
    - Registry
- Arguments for constructor (appear as "Parameters" in html docs):
    - same order as appear in constructor
    - last ones are always (in order): params, name, prefs  (with boilerplate descriptions)
    - brief description, with pointers to corresponding attribute for more detailed information.
- Attributes:
    - first line of each is: "attribute name : type : default <value>""
    - full description of all externally-accessible attributes, inclding use and constraints on values;
    - organized as close as possible to order of specification in constructor, instantiation, and/or execution;
    - last two are always (in order) name and prefs (with boilerplate descriptions)

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
