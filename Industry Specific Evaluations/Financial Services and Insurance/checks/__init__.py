"""Deterministic checkers for the FSI evaluation notebooks.

Each scenario owns one module:

* ``checks.servicing``: Scenario 1, authenticated customer-servicing agent.

Checkers read the inspectable state store that ``fsi_eval.systems_of_record``
records during a conversation, plus the user-visible response text. They never
call a model.
"""
