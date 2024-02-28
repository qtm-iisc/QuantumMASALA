# README for Generating Documentation for Quantum MASALA

- To assemble the source docs, use `sphinx-apidoc -e --ext-mathjax -o ./source -d 4 ../src/qtm/`
- To build the docs using autodoc, use `sphinx-build -b html ./source ./build`