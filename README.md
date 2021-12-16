# Online Marketplace Simulation: A Testbed for Self-Learning Agents

![Coverage-Badge](/src/tests/coverage.svg)
![Docstring-Coverage](/src/tests/docstrings.svg)
![CI](https://github.com/hpi-epic/BP2021/actions/workflows/CI.yml/badge.svg)

Working repository in context of the bachelorproject "Online Marketplace Simulation: A Testbed for Self-Learning Agents" at the research group Enterprise Platform and Integration Concepts.

The goal of the project is to develop a universal simulation platform for markets with varying numbers of merchants. Being able to run various market simulations is highly relevant for many firms such as SAP and its partners. As the platform is designed as a tool to support evaluation and research, aspects like configurability and ease of use are crucial. While the technology stack is left open for now, high compatibility to common simulation APIs (such as Gym, TF-Agents) is required.
For more complex setups, communication protocols between different agents might have to be implemented as well.

The simulation should cover the interaction between customers and particularly competing merchants, including self-learning agents and their rule-based opponents. While the focus can be put on several different aspects, an adjustable customer behavior model (which determines each participant!s sales) has to be developed. The platform should generate sales and interaction data for each of the merchants, which can then in turn be fed to the self-learning agents. Monitoring tools are required to analyze each agent!s policy and their effects on the overall market. With the help of such simulations, we seek to study the competitiveness of self-adapting pricing tools and their long-term impact on market competitors and customers.

## Installing dependencies

If you have not yet done so, install Anaconda and run the following command to create an environment and install the required packages from the `environment.yml`:

```console
conda env create -n your_venv_name
```

To activate your created environment use:

```console
conda activate your_venv_name
```

If you have a Nvidia GPU, consider installing cuda to get better training performance:
Note that depending on your specific GPU you might need to change the cudatoolkit version.

```console
conda uninstall pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
```

To update an existing environment with the needed packages run the following command:

```console
conda env update -n your_venv_name
```

If version numbers have changed in the `environment.yml` it can happen that conda finds conflicts and tries resolving them without succeeding. In this case you may need to reinstall the packages. Deactivate your environment before proceeding, otherwise conda cannot perform the commands.

```console
conda uninstall -n your_venv_name --all
conda env update -n your_venv_name
```

This will first uninstall all packages and then re-install them from the `environment.yml`.

## Using Pytest locally

[Pytest documentation](https://docs.pytest.org/en/latest/index.html)

If you want to run tests locally you can do so in a few ways:

- Simply running all tests:

```console
pytest
```

- Running all tests with increased verbosity:

```console
pytest -v
```

### Coverage

[Coverage.py Documentation](https://coverage.readthedocs.io/en/6.1.2/)

If you want to know/export the current test coverage use these commands:

- Run all tests, collect coverage info, write it to the coverage.json (for source code analysis of coverage) and update the coverage.svg badge:

```console
coverage run --source=. -m pytest
coverage json
coverage-badge -f -o ./src/tests/coverage.svg
```

- See coverage report locally:

```console
coverage report
```

## Pre-commit

[Pre-commit documentation](https://pre-commit.com/)

We are using `pre-commit` to lint our files before committing. Pre-commit itself should already have been installed through the `environment.yml`. Initialize pre-commit using

```console
pre-commit install
```

To circumvent possible errors caused later on, run pre-commit once with the following command:

```console
pre-commit run --all-files
```

which will install the needed environment.

### Pre-commit Troubleshooting

If you get the following error:

```console
Git: Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases
```

while trying to commit, the cause is most likely `pre-commit` trying to access a Python version not in your venv.

Solution: Check the App execution Aliases, and if no Python version is present, install it from the Microsoft Store. You do not need to disable the alias.

---

If you get an error saying that the `_sqlite3`-module is missing, you are missing the `sqlite3.dll` and `sqlite3.def` files.

Solution: Go to <https://www.sqlite.org/download.html> to download the `sqlite3.dll` and `sqlite3.def` files and drop them into the following folder:

```console
C:\Users\your_username\anaconda3\envs\your_venv_name\DLLs
```
