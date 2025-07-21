# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
import inspect
import os
import sys
from pathlib import Path

import rojak  # for linkcode_resolve

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
    "sphinx.ext.linkcode",  # Add external links to source code
    "sphinx_design",  # For pydata theme support
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
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}

# Allow markdown files to be recognised
source_suffix = [".rst", ".md"]
# Recommended syntax extension for myst_parser for markdown docs
# See https://sphinx-design.readthedocs.io/en/pydata-theme/get_started.html
myst_enable_extensions = ["colon_fence"]

SOURCE_BASE_URL: str = "https://github.com/ImperialCollegeLondon/rojak/blob"


# For links to source code on GitHub
# Modified from numpy's and pandas conf.py
# See https://github.com/numpy/numpy/blob/d02611ad99488637b48f4be203f297ea7b29c95d/doc/source/conf.py#L532
# See https://github.com/pandas-dev/pandas/blob/6a6a1bab4e0dccddf0c2e241c0add138e75d4a84/doc/source/conf.py#L645
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # FROM NUMPY: code doesn't have decorators atm
    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    # try:
    #     unwrap = inspect.unwrap
    # except AttributeError:
    #     pass
    # else:
    #     obj = unwrap(obj)

    # FROM PANDAS: numpy as some logic that looks specific to numpy itself
    try:
        # Blindly trust Pandas devs and get pyright to ignore
        fn = inspect.getsourcefile(inspect.unwrap(obj))  # pyright: ignore[reportArgumentType]
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    # Blindly trust Pandas devs and get pyright to ignore
    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""  # pyright: ignore[reportPossiblyUnboundVariable]
    fn = os.path.relpath(fn, start=Path(rojak.__file__).parent)

    return f"{SOURCE_BASE_URL}/master/src/rojak/{fn}{linespec}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# Path to static files and extra styling
html_static_path = ["_static"]
html_css_files = ["style.css"]

# See https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html#icon-link-shortcuts
html_theme_options = {"github_url": "https://github.com/ImperialCollegeLondon/rojak"}
