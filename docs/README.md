# README for Generating Documentation for Quantum MASALA

- To assemble the source docs, use 
```bash
sphinx-apidoc -e -E --ext-autodoc --ext-mathjax  -o ./source/rst_files -d 2 ../src/qtm/
```
- To build the docs using autodoc, use 
```bash
cp source/*.rst source/rst_files; cp source/conf.py source/rst_files; sphinx-build -b html ./source/rst_files ./build
```
- Start navigation from `index.html` in the `build` directory.

## Required libraries

```bash
python -m pip install sphinx sphinx-rtd-theme sphinx-math-dollar
```