# README for Generating Documentation for Quantum MASALA

We use Sphinx to compile documentation from the docstrings. The documentation is available on 'Read the Docs' at \url{https://quantummasala.readthedocs.io/en/latest/}, and can also be compiled locally.

To compile the documentation locally using Sphinx, install the following packages:
```
python -m pip install sphinx sphinx-rtd-theme sphinx-math-dollar
```

Compile the reStructuredText documentation by running the following from the root directory:
```
sphinx-apidoc -e -E --ext-autodoc --ext-mathjax -o docs/ -d 2 src/qtm/
```

Build the documentation in HTML format by running the following from the root directory:
```
sphinx-build -b html ./docs ./docs/build
```