# DEVELOPER GUIDE

This is a getting started guide for developers who would like to contribute to Quantum MASALA.

## Setting up the development environment

We use Git for version control, and GitHub for hosting our code. If you are not familiar with Git, you can learn more about it [here](https://github.com/git-guides).

### Fork the repository

The first step is to fork the Quantum MASALA repository. You can do this by running the following command:

```bash
git clone https://github.com/qtm-iisc/QuantumMASALA.git
```

### Install the dependencies
 
Please follow the instructions in the [INSTALL_BUILD.md](INSTALL_BUILD.md) and the [INSTALL.md](INSTALL.md) files to install the necessary dependencies. The editable pip install flag `-e` will enable the developers to make changes in the code, without having to re-install the `qtm` package. The full build process is described in the INSTALL_BUILD.md file is recommended for developers, to ensure that they can run the performance tests and benchmarks before submitting a pull request.

## Working on a new feature or bug fix

<!-- Credit for the instructions: Octopus code documentation. -->
Before starting to implement some new feature, we recommend you create a new issue on github explaining what you intent to do. The next step is to create a branch where to implement some new feature.

```bash
git checkout main
git checkout -b feature_name_branch
```

Before proceeding to commit your changes, please make sure that you are following the coding standards of the project. You can check the coding standards in the coding standards section of this document.

You can commit your modification in two steps:
```bash
git add modified_or_new_files
git commit -m "A message explaining what you did"
```

Make sure that you give a meaningful description of your changes to help other developers understand what you did. It might be a good idea to have a look at the changes that other developers made to see some examples of their messages.


To share your work with the other developers, you need to send your changes to the main repository. This is done with the push command. The first time you push your branch, you need to tell git where to send your changes. In this case you want to send them to the origin , which is the default name of the original repository you cloned

```bash
git push -u origin feature_name_branch
```

Git will remember this, so the next time you push your branch you will only need to do
    
```bash
git push
```

## Coding standards

### Testing

We use `pytest` for testing. The tests are located in the `tests` directory. To install `pytest`, you can run:

```bash
pip install pytest
```

To run the tests, you can use the following command:

```bash
pytest
```

You can also run the tests individually by specifying the test file or test function. For example:

```bash
pytest tests/system_tests/test_dft_si.py
```

This will run all the tests in the `tests` directory.

Please ensure that all the tests pass before submitting a pull request. Further, if you are adding a new feature, please add tests for the feature.

### Performance benchmarks

If you are working on a performance-critical part of the code, or making any other major changes, we recommend running the performance benchmarks to ensure that your changes do not degrade the performance. To run the performance benchmarks, please follow the instructions in the [benchmarks](benchmarks) directory.


### Documentation

We use `sphinx` for documentation. To build the documentation, see the instructions in the [docs](docs) directory.


### Code Formatting

We use `black` for code formatting. Black is [PEP 8](https://pep8.org/) compliant and will automatically format your code to adhere to the PEP 8 style guide.

To install `black`, you can run:

```bash
pip install black
```

To ensure your code is properly formatted, you can run:

```bash
black src/qtm
```

This will format all the Python files in the current directory and its subdirectories according to the `black` code style.

