# 1. Online Marketplace Simulation: A Testbed for Self-Learning Agents

![CI](https://github.com/hpi-epic/BP2021/actions/workflows/CI.yml/badge.svg)
![Coverage-Badge](/badges/coverage.svg)
![Docstring-Coverage](/badges/docstring_coverage.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

- [1. Online Marketplace Simulation: A Testbed for Self-Learning Agents](#1-online-marketplace-simulation-a-testbed-for-self-learning-agents)
	- [1.1. Installing dependencies](#11-installing-dependencies)
		- [1.1.1 Dependency Installation Troubleshooting](#111-dependency-installation-troubleshooting)
	- [1.2. The `AlphaBusiness` package](#12-the-alphabusiness-package)
	- [1.3. Testing](#13-testing)
		- [1.3.1. Pytest](#131-pytest)
		- [1.3.2.](#132)
		- [1.3.3. Coverage](#133-coverage)
	- [1.4. Pre-commit](#14-pre-commit)
		- [1.4.1. Interrogate](#141-interrogate)
		- [1.4.2. Pre-commit Troubleshooting](#142-pre-commit-troubleshooting)
	- [1.5. Networking Scenario](#15-networking-scenario)
		- [1.5.1. Docker](#151-docker)
			- [1.5.1.1. Docker-API](#1511-docker-api)
			- [1.5.1.2. Using Docker natively](#1512-using-docker-natively)
		- [1.5.2. Webserver](#152-webserver)
		- [1.5.3. Docker API](#153-docker-api)

Working repository in context of the bachelorproject "Online Marketplace Simulation: A Testbed for Self-Learning Agents" at the research group Enterprise Platform and Integration Concepts.

The goal of the project is to develop a universal simulation platform for markets with varying numbers of merchants. Being able to run various market simulations is highly relevant for many firms such as SAP and its partners. As the platform is designed as a tool to support evaluation and research, aspects like configurability and ease of use are crucial. While the technology stack is left open for now, high compatibility to common simulation APIs (such as Gym, TF-Agents) is required.
For more complex setups, communication protocols between different agents might have to be implemented as well.

The simulation should cover the interaction between customers and particularly competing merchants, including self-learning agents and their rule-based opponents. While the focus can be put on several different aspects, an adjustable customer behavior model (which determines each participant!s sales) has to be developed. The platform should generate sales and interaction data for each of the merchants, which can then in turn be fed to the self-learning agents. Monitoring tools are required to analyze each agent!s policy and their effects on the overall market. With the help of such simulations, we seek to study the competitiveness of self-adapting pricing tools and their long-term impact on market competitors and customers.

## 1.1. Installing dependencies

We are using both pip and conda to install our dependencies. The difference between the two is that we install all dependencies we need to run the core functionality of our project using pip, and all other dependencies (such as pytest or django) with conda. This allows us to keep our docker containers small by only installing pip dependencies there.

If you have not yet done so, install Anaconda and run the following command to create an environment and install the required packages from the `environment.yml`:

```bash
conda env create -n your_venv_name
```

To activate your created environment use:

```bash
conda activate your_venv_name
```

Additionally, pip dependencies need to be installed using the following command. Make sure you activate your conda environment first!

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

The `-f` flag tells pip to look at the specified URL for links to archives. This is needed to find the pytorch version with cuda support (`torch==1.10.2+cu113`)

To update an existing environment with the needed packages run the following command:

```bash
conda env update -n your_venv_name
```

If version numbers have changed in the `environment.yml` it can happen that conda finds conflicts and tries resolving them without succeeding. In this case you may need to reinstall the packages. Deactivate your environment before proceeding, otherwise conda cannot perform the commands.

```bash
conda uninstall -n your_venv_name --all
conda env update -n your_venv_name
```

This will first uninstall all packages and then re-install them from the `environment.yml`.

### 1.1.1 Dependency Installation Troubleshooting

If you get the following error message when trying to access the docker SDK (e.g. by starting the API through `docker/app.py`):

```bash
docker.errors.DockerException: Install pypiwin32 package to enable npipe:// support
```

you have to run the `pywin32_postinstall.py` script. To do so, run the following command:

```bash
python Path/To/Anaconda3/Scripts/pywin32_postinstall.py -install
```

## 1.2. The `AlphaBusiness` package

You may have noticed the following line in the `requirements.txt`:

```bash
-e ./src
```

This installs the `src` folder (and its subdirectories) as a local pip package. The `-e` flag indicates to pip that the package should be installed in an editable state, meaning that any changes to `.py` files in the package will be integrated into the package immediately (meaning no re-install is necessary). In order to install the package, pip looks into the `setup.py` file where name, version and packages of the new package are set.
You can confirm that the install was successfull if there is a folder called `AlphaBusiness.egg-info` within the `src`-directory, or by checking `pip freeze` for the following line:

```bash
-e git+https://github.com/hpi-epic/BP2021.git@da8868467690a1300ff4e11245417ec384aae15b#egg=AlphaBusiness&subdirectory=src
```

If you do not see the `AlphaBusiness`-package in the resulting list, perform the following command while in the top-level-folder of the repository (e.g. `BP2021`):

```bash
pip install -e ./src
```

Check `pip freeze` again to make sure the package was installed.

Installing our project as a package enables us to perform [relative imports](https://realpython.com/absolute-vs-relative-python-imports/) from within subdirectories to parent directories. The most prominent example of this would be importing the tested files from within the test-files in the `tests/` subdirectory.
Package installation adapted from [this post](https://stackoverflow.com/a/50194143).

## 1.3. Testing

### 1.3.1. Pytest

[Pytest documentation](https://docs.pytest.org/en/latest/index.html)

If you want to run tests locally you can do so in a few ways:

- Simply running all tests:

```bash
pytest
```

- Running all tests with increased verbosity:

```bash
pytest -v
```

### 1.3.2.
To run tests you have written for the Django webserver go into the *webserver* folder and run

```bash
python3 manage.py test
```

### 1.3.3. Coverage

[Coverage.py Documentation](https://coverage.readthedocs.io/en/6.1.2/)

If you want to know/export the current test coverage use these commands:

- Run all tests, collect coverage info, write it to the coverage.json (for source code analysis of coverage) and update the coverage.svg badge:

```bash
coverage run --source=. -m pytest
coverage json
coverage-badge -f -o ./badges/coverage.svg
```

- See coverage report locally:

```bash
coverage report
```

## 1.4. Pre-commit

[Pre-commit documentation](https://pre-commit.com/)

We are using `pre-commit` to lint our files before committing. Pre-commit itself should already have been installed through the `environment.yml`. Initialize pre-commit using

```bash
pre-commit install
```

To circumvent possible errors caused later on, run pre-commit once with the following command:

```bash
pre-commit run --all-files
```

which will install the needed environment.

### 1.4.1. Interrogate

[Interrogate documentation](https://interrogate.readthedocs.io/en/latest/)

We use Interrogate to monitor our docstring coverage. Interrogate is automatically run with Pre-commit, but the badge can only be updated manually.
To update the badge, modify the `pre-commit-config.yml` file by swapping the following line:

```yml
args: [-v, --ignore-init-method, --ignore-module, --exclude=./src/tests, --exclude=./webserver, --fail-under=50]
```
with
```yml
args: [-v, --ignore-init-method, --ignore-module, --exclude=./src/tests, --exclude=./webserver, --fail-under=50, --generate-badge=./badges/docstring_coverage.svg, --badge-style=flat]
```


### 1.4.2. Pre-commit Troubleshooting

If you get the following error:

```bash
Git: Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases
```

while trying to commit, the cause is most likely `pre-commit` trying to access a Python version not in your venv.

Solution: Check the App execution Aliases, and if no Python version is present, install it from the Microsoft Store. You do not need to disable the alias.

---

If you get an error saying that the `_sqlite3`-module is missing, you are missing the `sqlite3.dll` and `sqlite3.def` files.

Solution: Go to the [SQLite Download Page](https://www.sqlite.org/download.html) to download the `sqlite3.dll` and `sqlite3.def` files and drop them into the anaconda installation folder:

```PATH
C:\Users\your_username\anaconda3\envs\your_venv_name\DLLs
```

## 1.5. Networking Scenario

### 1.5.1. Docker

[Docker SDK documentation](https://docker-py.readthedocs.io/en/stable/index.html)

To use docker, first install it on your machine. Afterwards, you can build the images used in our repository using the following command:

```bash
python3 ./docker/docker_manager.py
```

#### 1.5.1.1. Docker-API

We recommend interacting with Docker using the Webserver as outlined in the [Webserver](#webserver) section. If you still want to use Docker from you commandline, refer to [Using Docker natively](#using-docker-natively).

#### 1.5.1.2. Using Docker natively

This command will create an image for each command that can be executed in a docker container. Building the images may take a while, it is about 5GB in size. To see all current images on your system use:

```bash
docker images
```

You can create and run a container for an image using the following command:

```bash
docker run IMAGE_ID
```

At any point you can list all current containers with:

```bash
docker ps -a
```

You can stop a container using:

```bash
docker stop CONTAINER_ID
```

And remove it with:

```bash
docker remove CONTAINER_ID
```

### 1.5.2. Webserver

We provide a Django Webserver with a simple user interface to manage the docker container.
To start the webserver on `127.0.0.1:2709` go to `/webserver` and start the server by using the following command

```bash
python3 ./manage.py runserver 2709
```

When you add more fields to the database model or you change existing fields, you need to run

```bash
python3 ./manage.py makemigrations
```

Before starting the server you might need to apply any pending migrations using

```bash
python3 ./manage.py migrate
```

To run tests you have written for the Django webserver go into the _webserver_ folder and run

```bash
python manage.py test
```

### 1.5.3. Docker API

There is a RESTful API written with the python libary FastAPI for communicating with docker containers that can be found in `/docker`

The API needs to run on `127.0.0.1:8000`. To start the API go to `/docker` and run

```bash
uvicorn app:app --reload 
```

Don't use `--reload` when deploying in production.

You can just run the `app.py` with python from the docker folder as well.
