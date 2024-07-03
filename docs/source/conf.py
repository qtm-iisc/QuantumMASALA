# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add to path, the source code dir relative to this file
sys.path.insert(0, os.path.abspath('../../src/'))  # Adjust the path as needed



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Quantum MASALA'
copyright = '2024, S. Shri Hari, Agrim Sharma, Manish Jain'
author = 'S. Shri Hari, Agrim Sharma, Manish Jain'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',         # Core library for html generation from docstrings
              'sphinx.ext.napoleon',        # Support for NumPy and Google style docstrings
              'sphinx_math_dollar',         # Allow $..$ for inline math
              'sphinx.ext.mathjax',         # Add MathJax for rendering math
              'sphinx.ext.autosummary',     # Create neat summary tables 
              ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Set the napoleon options to parse NumPy style docstrings
napoleon_numpy_docstring = True
napoleon_use_param = True  # This option is personal preference, adjust as needed


# -- Create a separate page for each submodule -------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_default_flags

autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'special-members']
autodoc_member_order = 'bysource'
autodoc_typehints = "description"
autosummary_generate = True


mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-cq.js'

# Use dollar signs for math
math_dollar = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_material'      # pip install sphinx_material
html_theme = 'sphinx_rtd_theme'       # pip install sphinx_rtd_theme
# html_static_path = ['_static']


mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}


# MathJax options
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'  # CDN link to MathJax

# Optional: If you have other LaTeX in your docs, configure LaTeX to HTML translation
latex_elements = {
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{amssymb}
    ''',
}
