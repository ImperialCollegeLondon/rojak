# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rojak"
copyright = "2025, Hui Ling Wong"
author = "Hui Ling Wong"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.apidoc",  # Generate API documentation from Python packages
    "sphinx.ext.autosummary",  # Generate autodoc summaries
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "myst_parser",  # Markdown support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mapping to other project documentation
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Allow markdown files to be recognised
source_suffix = [".rst", ".md"]

# Related to sphinx.ext.autodoc
apidoc_modules = [{"path": "../src/rojak", "destination": "api/"}]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
