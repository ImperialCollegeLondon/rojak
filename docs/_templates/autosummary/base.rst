{# Modified from: https://github.com/sphinx-doc/sphinx/blob/a15c149a607a1dcbc07e0058108194726d382d9f/sphinx/ext/autosummary/templates/autosummary/base.rst #}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
