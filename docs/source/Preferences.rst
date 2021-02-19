Preferences
===========

Standard prefereces:

- verbose (bool): enables/disables reporting of (non-exception) warnings and system function

- paramValidation (bool): enables/disables run-time validation of the execute method of a Function object

- reportOutput ([bool, str]): enables/disables reporting execution of `Component`\'s `execute <Component_Execution>`
  method to console:
    - ``True``: prints record of execution, including the input and output of the Component;
    - 'params' or 'parameters': includes report of the Component's `parameter <Parameters>` values.
    - 'terse': (Composition only) restricts output to just a listing of `trials <TimeScale.TRIAL>`, `time_steps
      <TimeScale.TIME_STEP>` and `nodes <Composition_Nodes>` executed.

- log (bool): sets LogCondition for a given Component

- functionRunTimeParams (Modulation): uses run-time params to modulate execute method params

[ADDITIONAL DOCUMENTATION COMING...]


.. .. automodule:: psyneulink.core.globals.preferences
   :members:
   :exclude-members: Parameters, PreferenceLevel, PreferenceSetError, PreferenceEntry, PreferenceSetRegistry
