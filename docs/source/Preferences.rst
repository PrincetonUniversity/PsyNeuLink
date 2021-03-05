Preferences
===========

Standard prefereces:

- verbosePref (bool): enables/disables reporting of (non-exception) warnings and system function

- paramValidationPref (bool): enables/disables run-time validation of the execute method of a Function object

- reportOutputPref ([bool, str]): enables/disables reporting execution of `Component`\'s `execute <Component_Execution>`
  method to console and/or PsyNeuLinkView:
    - ``True``: prints record of execution, including the input and output of the Component;
    - *TERSE*: restricts output to just a statement that the Component executed;
    - 'params' or 'parameters': includes report of the Component's `parameter <Parameters>` values.

- logPref (bool): sets LogCondition for a given Component

- functionRunTimeParamsPref (Modulation): uses run-time params to modulate execute method params

[ADDITIONAL DOCUMENTATION COMING...]


.. .. automodule:: psyneulink.core.globals.preferences
   :members:
   :exclude-members: Parameters, PreferenceLevel, PreferenceSetError, PreferenceEntry, PreferenceSetRegistry
