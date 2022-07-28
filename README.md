# Online Marketplace Simulation: A Testbed for Self-Learning Agents

![CI](https://github.com/hpi-epic/BP2021/actions/workflows/CI.yml/badge.svg)
![Coverage-Badge](/badges/coverage.svg)
![Docstring-Coverage](/badges/docstring_coverage.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Working repository in context of the bachelor's project "*Online Marketplace Simulation: A Testbed for Self-Learning Agents*" at the *Enterprise Platform and Integration Concepts* ([@hpi-epic](https://github.com/hpi-epic), [epic.hpi.de](https://hpi.de/plattner/home.html)) research group of the Hasso Plattner Institute.

During the project a simulation framework, *`recommerce`*, was built.
The framework can be used to simulate online *recommerce* marketplaces, which are more complex than traditional linear economy models.
Using these simulated marketplaces, various reinforcement learning algorithms can be used to train so-called *agents* to set prices that optimize profit on these markets.
These agents can be trained against a multitude of pre-implemented rule based vendors, on a number of different marketplaces.

After training, the produced models can be further monitored and analyzed using an extensive number of monitoring tools.
For more information beyond installation and the Quick Start Guide contained in this `README`, please refer to our [wiki](https://github.com/hpi-epic/BP2021/wiki).

- [Quick Start Guide](#quick-start-guide)
- [Installing dependencies](#installing-dependencies)
- [Installing the `recommerce` package](#installing-the-recommerce-package)

## Quick Start Guide

If you have not yet done so, first install the `recommerce` package by following [installing dependencies](#installing-dependencies) and [installing the `recommerce` package](#installing-the-recommerce-package).

At any point after the installation has completed, use

```terminal
recommerce --help
```

in your terminal to see the usage options of the package.
The guide below will help you set up the framework and get started with a first experiment.

The `recommerce` package requires users to provide it with a datapath, which is where the package will look for configuration files and write output files, such as statistics and diagrams.
During installation of the package, the datapath was set to the current working directory.
If you want to modify the datapath, you can use the following command:

```terminal
recommerce --datapath "<datapath>"
```

where <datapath> is a valid path from which `recommerce` will read and write data.

You should see the following message indicating your path is valid:

```terminal
Data will be read from and saved to "<datapath>"
```

You can check the currently set datapath at any point by running

```terminal
recommerce --get-datapath
```

To start an experiment, `recommerce` requires you to provide configuration files, which contain the necessary information to set up the simulation.
You can either write those files yourself or use the following command to have `recommerce` copy over default files which you can immediately use or modify as you wish:

*WARNING*: By using the following command, any files with the same names as the default data will be overwritten, so use with caution!
To get the data in a folder called `default_data` use the flag `--get-defaults` instead.

```terminal
recommerce --get-defaults-unpack
```

The `-unpack` part of the flag makes sure that the default files are not just stored in the `default_data` folder in your datapath, but unpacked in a way that `recommerce` can immediately find them, by storing configuration files and pre-trained models in the `configuration_files` and `data` directories respectively.

Now you are ready to run your first exampleprinter session:

```terminal
recommerce -c exampleprinter
```

If you are using the default configuration files provided by `recommerce --get-defaults`, this exampleprinter run will also create an in-depth HTML-Slideshow of the run in the `results/exampleprinter` folder within your datapath.

## Installing dependencies

We are using both [pip](https://pip.pypa.io/en/stable/index.html) and [`anaconda`](https://www.anaconda.com/) to install our dependencies.
The difference between the two is that we install all dependencies we need to run the core functionality of our project using `pip`, and all other dependencies (such as `pytest` or `django`) using `anaconda`.
This allows us to have a clear distinction between the purposes of the different packages, while also keeping our docker containers small by only installing pip dependencies there.
You can read up more about our usage of docker [in the wiki](https://github.com/hpi-epic/BP2021/wiki/Developer-guides-%E2%80%93-Docker-&-UI).

For the purpose of staying user-centered, this Quick Start Guide will only include installation instructions for the `pip` dependencies, through the use of our `recommerce` package, described in the next section.
If you plan on not only using the framework but also continue its development, please take a look at the installation instructions in the [developer guide](https://github.com/hpi-epic/BP2021/wiki/Developer-guides-%E2%80%93-Installation) first, to ensure that all dependencies are installed correctly using an `anaconda` virtual environment (though we recommend installing the `recommerce` package in a virtual environment in any case).

## Installing the `recommerce` package

*NOTE: If you want to continue development of the simulation framework, please read the installation instructions in the [developer guide](https://github.com/hpi-epic/BP2021/wiki/Developer-guides-%E2%80%93-Installation) **first**!*

Before proceeding, please inform yourself on whether or not your device supports `cuda`.
Take a look at [this](https://developer.nvidia.com/cuda-gpus) resource provided by NVIDIA for help on the topic.
This decides if you should install our project with `cuda` support, or without, which comes down to the specific version of the *torch* dependency that will be installed.

*Depending on your preferences, you may first want to create a dedicated virtual environment for the `recommerce` package (recommended).*

If you want to continue development of the framework, meaning you plan on changing the code, insert the `-e` flag after `pip install` in the following commands to install the package in an editable state.
This results in the packages not being directly written to where pip dependencies usually would, but only a "link" to you current working directory being created.

If your device supports `cuda` and you want to utilize its capabilities, use the following command within the project directory:

```terminal
pip install .[gpu] -f https://download.pytorch.org/whl/torch_stable.html
```

Otherwise, to install without `cuda` support, use:

```terminal
pip install .[cpu]
```

This installs the `recommerce` project folder (and its subdirectories) as a local `pip` package (when using `anaconda`, this will be `Path/To/anaconda3/envs/your_venv_name/Lib/site-packages`).
In order to install the package, we use `setuptools`, which uses the `setup.py`, `setup.cfg` and `pyproject.toml` files located in the root directory of the project.
The `setup.cfg` file includes all necessary metadata needed to correctly install the project.

You can confirm that the installation was successful by checking

```terminal
recommerce --version
```
