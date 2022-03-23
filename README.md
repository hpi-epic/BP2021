# Online Marketplace Simulation: A Testbed for Self-Learning Agents  <!-- omit in toc -->

![CI](https://github.com/hpi-epic/BP2021/actions/workflows/CI.yml/badge.svg)
![Coverage-Badge](/badges/coverage.svg)
![Docstring-Coverage](/badges/docstring_coverage.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

- [1. Using the `recommerce` package](#1-using-the-recommerce-package)
	- [1.1. Quick-Start guide](#11-quick-start-guide)
- [2. Installing dependencies](#2-installing-dependencies)
	- [2.1. Dependency Installation Troubleshooting](#21-dependency-installation-troubleshooting)
- [3. Installing the `Recommerce` package](#3-installing-the-recommerce-package)
- [4. Testing](#4-testing)
	- [4.1. Pytest](#41-pytest)
	- [4.2. Webserver Tests](#42-webserver-tests)
	- [4.3. Coverage](#43-coverage)
- [5. Pre-commit](#5-pre-commit)
	- [5.1. Interrogate](#51-interrogate)
	- [5.2. Pre-commit Troubleshooting](#52-pre-commit-troubleshooting)
- [6. Networking Scenario](#6-networking-scenario)
	- [6.1. Docker](#61-docker)
		- [6.1.1. Docker-API](#611-docker-api)
		- [6.1.2. Using Docker natively](#612-using-docker-natively)
	- [6.2. Webserver](#62-webserver)
	- [6.3. Docker API](#63-docker-api)
- [7. Tensorboard](#7-tensorboard)

Working repository in context of the bachelorproject "Online Marketplace Simulation: A Testbed for Self-Learning Agents" at the research group Enterprise Platform and Integration Concepts.

The goal of the project is to develop a universal simulation platform for markets with varying numbers of merchants. Being able to run various market simulations is highly relevant for many firms such as SAP and its partners. As the platform is designed as a tool to support evaluation and research, aspects like configurability and ease of use are crucial. While the technology stack is left open for now, high compatibility to common simulation APIs (such as Gym, TF-Agents) is required.
For more complex setups, communication protocols between different agents might have to be implemented as well.

The simulation should cover the interaction between customers and particularly competing merchants, including self-learning agents and their rule-based opponents. While the focus can be put on several different aspects, an adjustable customer behavior model (which determines each participant!s sales) has to be developed. The platform should generate sales and interaction data for each of the merchants, which can then in turn be fed to the self-learning agents. Monitoring tools are required to analyze each agent!s policy and their effects on the overall market. With the help of such simulations, we seek to study the competitiveness of self-adapting pricing tools and their long-term impact on market competitors and customers.

## 1. Using the `recommerce` package

If you have not yet done so, first install the package by following [Installing dependencies](#2-installing-dependencies) and [Installing the `recommerce` package](#3-installing-the-recommerce-package).

At any point, use

```terminal
recommerce --help
```

to see the usage options of the package.

### 1.1. Quick-Start guide

After installing, you will need to provide `recommerce` with a datapath, which is where `recommerce` will look for configuration files and write output files, such as trained models. Do so by using

```terminal
recommerce --datapath "your_preferred_path"
```

You should see the following message indicating your path is valid:

```terminal
Data will be read from and saved to "your_preferred_path"
```

To start your first training or exampleprinter session, `recommerce` requires you to provide two configuration files. You can either write those yourself or use the following command to have `recommerce` copy over default files which you can immediately use or modify as you wish.

*WARNING*: By using the following command, any files with the same names as the default data will be overwritten, so use with caution! To get the data in a folder called `default_data` use the flag `--get-defaults` instead.

```terminal
recommerce --get-defaults-unpack
```

The `-unpack` part flag makes sure that the default files are not just stored in the `default-data` folder in your datapath, but unpacked in a way that `recommerce` can find them.

Now you are ready to run your first exampleprinter session:

```terminal
recommerce -c exampleprinter
```

If you are using the default configuration files provided by `recommerce --get-defaults`, this exampleprinter run will create an in-depth HTML-Slideshow of the run in the `results/exampleprinter` folder within your datapath.

## 2. Installing dependencies

We are using both pip and conda to install our dependencies. The difference between the two is that we install all dependencies we need to run the core functionality of our project using pip, and all other dependencies (such as pytest or django) with conda. This allows us to keep our docker containers small by only installing pip dependencies there.

If you have not yet done so, install Anaconda and run the following command to create an environment and install the required packages from the `environment.yml`:

```terminal
conda env create -n your_venv_name
```

To activate your created environment use:

```terminal
conda activate your_venv_name
```

To update an existing environment with the needed packages run the following command:

```terminal
conda env update -n your_venv_name
```

If version numbers have changed in the `environment.yml` it can happen that conda finds conflicts and tries resolving them without succeeding. In this case you may need to reinstall the packages. Deactivate your environment before proceeding, otherwise conda cannot perform the commands.

```terminal
conda uninstall -n your_venv_name --all
conda env update -n your_venv_name
```

This will first uninstall all packages and then re-install them from the `environment.yml`.

### 2.1. Dependency Installation Troubleshooting

If you get the following error message when trying to access the docker SDK (e.g. by starting the API through `docker/app.py`):

```terminal
docker.errors.DockerException: Install pypiwin32 package to enable npipe:// support
```

you have to run the `pywin32_postinstall.py` script. To do so, run the following command:

```terminal
python Path/To/Anaconda3/Scripts/pywin32_postinstall.py -install
```

## 3. Installing the `Recommerce` package

In order to use our project, you must perform the following command:

```terminal
pip install -e .
```

This installs the `recommerce` folder (and its subdirectories) as a local pip package. The `-e` flag indicates to pip that the package should be installed in an editable state. This results in the packages not being directly written to where pip dependencies usually would, but only a "link" to you current working directory being created. In order to install the package, we use `setuptools`, which uses the `setup.py`, `setup.cfg` and `pyproject.toml` files located in the `recommerce` directory of the project. The `setup.cfg` file includes all necessary metadata needed to correctly install the project. If you want to properly install the project as a pip package, use the following command:

```terminal
pip install .
```

This will "copy" the packages into your pip-installation folder (when using conda, this will be `Path/To/anaconda3/envs/your_venv_name/Lib/site-packages`), meaning that any changes to your source-code will only be reflected when installing the package again, therefore you should not use that command if you plan on changing the code.

You can confirm that the installation was successfull by checking

```terminal
recommerce --version
```

Package installation was adapted from [this repository](https://github.com/mCodingLLC/SlapThatLikeButton-TestingStarterProject).

## 4. Testing

### 4.1. Pytest

[Pytest documentation](https://docs.pytest.org/en/latest/index.html)

If you want to run tests locally you can do so in a few ways:

- Simply running all tests:

```terminal
pytest
```

- Running all tests with increased verbosity:

```terminal
pytest -v
```

Additionally, if you have installed the `recommerce` package and the `tests`-folder is located in your `datapath`, you can use the following command to run the tests:

```terminal
recommerce --test
```

### 4.2. Webserver Tests

To run tests you have written for the Django webserver go into the *webserver* folder and run

```terminal
python3 ./manage.py test
```

### 4.3. Coverage

[Coverage.py Documentation](https://coverage.readthedocs.io/en/6.1.2/)

If you want to track the current test coverage use these commands:

- Run all tests, collect coverage info, write it to the coverage.json (for source code analysis of coverage) and update the coverage.svg badge:

```terminal
coverage run --source=. -m pytest
coverage json
coverage-badge -f -o ./badges/coverage.svg
```

- See coverage report locally:

```terminal
coverage report
```

## 5. Pre-commit

[Pre-commit documentation](https://pre-commit.com/)

We are using `pre-commit` to lint our files before committing. Pre-commit itself should already have been installed through the `environment.yml`. Initialize pre-commit using

```terminal
pre-commit install
```

To circumvent possible errors caused later on, run pre-commit once with the following command:

```terminal
pre-commit run --all-files
```

which will install the needed environment.

### 5.1. Interrogate

[Interrogate documentation](https://interrogate.readthedocs.io/en/latest/)

We use Interrogate to monitor our docstring coverage. It is being configured within the `pyproject.toml` in the root directory of the project. Interrogate is automatically run with Pre-commit, but the badge can only be updated manually.
To update the badge, modify the `pre-commit-config.yml` file by swapping the following line:

```yml
args: [--config=pyproject.toml]
```

with

```yml
args: [--config=pyproject.toml, --generate-badge=./badges/docstring_coverage.svg, --badge-style=flat]
```

### 5.2. Pre-commit Troubleshooting

If you get the following error:

```terminal
Git: Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases
```

while trying to commit, the cause is most likely `pre-commit` trying to access a Python version not in your venv.

Solution: Check the App execution Aliases, and if no Python version is present, install it from the Microsoft Store. You do not need to disable the alias.

---

If you get an error saying that the `_sqlite3`-module is missing, you are missing the `sqlite3.dll` and `sqlite3.def` files.

Solution: Go to the [SQLite Download Page](https://www.sqlite.org/download.html) to download the `sqlite3.dll` and `sqlite3.def` files and drop them into the anaconda installation folder:

```PATH
Path\To\anaconda3\envs\your_venv_name\DLLs
```

## 6. Networking Scenario

### 6.1. Docker

[Docker SDK documentation](https://docker-py.readthedocs.io/en/stable/index.html)

To use docker, first install it on your machine, for more convenience we recommend using [Docker Desktop](https://www.docker.com/products/docker-desktop/). Afterwards, you can build the images used in our repository using the following command:

```terminal
python3 ./docker/docker_manager.py
```

#### 6.1.1. Docker-API

We recommend interacting with Docker using the Webserver as outlined in the [Webserver](#62-webserver) section. If you still want to use Docker from you commandline, refer to [Using Docker natively](#612-using-docker-natively).

#### 6.1.2. Using Docker natively

This command will create an image for each command that can be executed in a docker container. Building the images may take a while, it is about 5GB in size. To see all current images on your system use:

```terminal
docker images
```

You can create and run a container for an image using the following command:

```terminal
docker run IMAGE_ID
```

At any point you can list all current containers with:

```terminal
docker ps -a
```

You can stop a container using:

```terminal
docker stop CONTAINER_ID
```

And remove it with:

```terminal
docker remove CONTAINER_ID
```

### 6.2. Webserver

We provide a Django Webserver with a simple user interface to manage the docker container.
To start the webserver on `127.0.0.1:2709` go to `/webserver` and start the server by using the following command

```terminal
python3 ./manage.py runserver 2709
```

When you add more fields to the database model or you change existing fields, you need to run

```terminal
python3 ./manage.py makemigrations
```

Before starting the server you might need to apply any pending migrations using

```terminal
python3 ./manage.py migrate
```

To run tests you have written for the Django webserver go into the *webserver* folder and run

```terminal
python3 ./manage.py test
```

We use bootstrap for layout stuff. When developing, you need to run `npm install` in the webserver directory.

### 6.3. Docker API

There is a RESTful API written with the python libary FastAPI for communicating with docker containers that can be found in `/docker`.

The API needs to run on `127.0.0.1:8000`. To start the API go to `/docker` and run

```terminal
uvicorn app:app --reload 
```

Don't use `--reload` when deploying in production.

You can just run the `app.py` with python from the docker folder as well.

## 7. Tensorboard

Tensorboard is used extensively to track parameters of the training as well as the agents and the market itself. It is started from the console by using:

```terminal
tensorboard serve --logdir your_data_path
```

The path specified can be changed to just include one subfolder of the runs folder to track just one of the experiments.
NOTE: It might not work with Safari, but Chrome does the job.

If you are using the webserver the yellow button opens the tensorboard, you might need to reload the page it redirects you to, because the tensorboard server in the container did not start fast enough, most times the browser will do this for you.
