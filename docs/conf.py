# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QuantumMASALA'
copyright = '2023, Shri Hari S, Agrim Sharma, Manish Jain'
author = 'Shri Hari S, Agrim Sharma, Manish Jain'
release = '0.9.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',  # For automatic generation of API documentation
    'sphinx.ext.intersphinx',  # For generating hyperlinks to other packages
    'sphinx.ext.napoleon',  # Support for NumPy-style docstrings
]

# Napoleon Config
napoleon_numpy_docstring = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
default_role = 'py:obj'

typehints_use_signature = True
typehints_use_signature_return = True
typehints_use_rtype = True

# AutoAPI Config
autoapi_type = 'python'
autoapi_dirs = ['../src/']
autoapi_options = [
    'members', 'undoc-members', 'private-members', 'show-inheritance',
    # 'show-module-summary',  # Disabled as it does not render properly for type-hints
    'special-members', 'imported-members',
]
autoapi_member_order = 'bysource'

# Intersphinx Config
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'cupy': ('https://docs.cupy.dev/en/stable/', None),
    'mpi4py': ('https://mpi4py.readthedocs.io/en/stable/', None),
    'pyfftw': ('https://pyfftw.readthedocs.io/en/latest/', None),
}


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
