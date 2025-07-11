# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "src").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rojak"
copyright = "2025, Hui Ling Wong"  # noqa: A001
author = "Hui Ling Wong"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.apidoc",  # Generate API documentation from Python packages
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",  # Links to other projects
    "sphinx.ext.autosummary",  # Generate autodoc summaries
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "myst_parser",  # Markdown support
    "jupyter_sphinx",  # Executes embedded code in a Jupyter kernel
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# This needs to be set despite the default being true
# See https://github.com/sphinx-doc/sphinx/issues/6800
autosummary_generate = True

autoclass_content = "both"
autodoc_member_order = "groupwise"
autodoc_typehints = "both"

# Mapping to other project documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
}

# Allow markdown files to be recognised
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"]
