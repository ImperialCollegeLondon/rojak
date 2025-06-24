{# https://github.com/sphinx-doc/sphinx/blob/a15c149a607a1dcbc07e0058108194726d382d9f/sphinx/ext/autosummary/templates/autosummary/module.rst #}

{% if name == "cli" %}
   {% set header_name = "CLI" %}
{% elif name == "datalib" %}
   {% set header_name = "Data Library" %}
{% else %}
   {% set header_name = name | title | replace("_", " ") | escape %}
{% endif %}

{{ (header_name | title | escape ~ " ``(" ~  fullname ~ ")`` ") | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree: ./
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree: ./
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %} {%- if modules %} .. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}
    {{ item }}

{%- endfor %} {% endif %} {%- endblock %}
