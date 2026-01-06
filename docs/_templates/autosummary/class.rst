{# Modified from: https://github.com/sphinx-doc/sphinx/blob/a15c149a607a1dcbc07e0058108194726d382d9f/sphinx/ext/autosummary/templates/autosummary/class.rst #}

{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
