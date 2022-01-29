# Online Marketplace Simulation: A Testbed for Self-Learning Agents

![CI](https://github.com/hpi-epic/BP2021/actions/workflows/CI.yml/badge.svg)
![Coverage-Badge](/badges/coverage.svg)
![Docstring-Coverage](/badges/docstring_coverage.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

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

Additionally, pip dependencies need to be installed using the following command. Make sure you activate your conda environment first!

```console
pip install -r requirements.txt
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

## The `AlphaBusiness` package

You may have noticed the following lines in the `requirements.txt`:
```
	-e ./src
```

This installs the `src` folder (and its subdirectories) as a local pip package. The `-e` flag indicates to pip that the package should be installed in an editable state, meaning that any changes to `.py` files in the package will be integrated into the package immediately (meaning no re-install is necessary). In order to install the package, pip looks into the `setup.py` file where name, version and packages of the new package are set.
You can confirm that the install was successfull if there is a folder called `AlphaBusiness.egg-info` within the `src`-directory, or by checking `pip freeze` for the following line: 

```
-e git+https://github.com/hpi-epic/BP2021.git@da8868467690a1300ff4e11245417ec384aae15b#egg=AlphaBusiness&subdirectory=src
```

If you do not see the `AlphaBusiness`-package in the resulting list, perform the following command while in the top-level-folder of the repository (e.g. `BP2021`):

```console
pip install -e ./src
```

Check `pip freeze` again to make sure the package was installed.

Installing our project as a package enables us to perform [relative imports](https://realpython.com/absolute-vs-relative-python-imports/) from within subdirectories to parent directories. The most prominent example of this would be importing the tested files from within the test-files in the `tests/` subdirectory.
Package installation adapted from [this post](https://stackoverflow.com/a/50194143).

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
coverage-badge -f -o ./badges/coverage.svg
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

```
Git: Python was not found; run without arguments to install from the Microsoft Store, 
or disable this shortcut from Settings > Manage App Execution Aliases
```

while trying to commit, the cause is most likely `pre-commit` trying to access a Python version not in your venv.

Solution: Check the App execution Aliases, and if no Python version is present, install it from the Microsoft Store. You do not need to disable the alias.

---

If you get an error saying that the `_sqlite3`-module is missing, you are missing the `sqlite3.dll` and `sqlite3.def` files.

Solution: Go to <https://www.sqlite.org/download.html> to download the `sqlite3.dll` and `sqlite3.def` files and drop them into the following folder:

```
C:\Users\your_username\anaconda3\envs\your_venv_name\DLLs
```

## Networking Scenario

### Docker

To use docker, please install it on your machine. If you did so, you can build an image with the following command:

```console
docker build . -t bp2021image
```

This probably will take a while (especially the first time, since the image is ~17GB in size). The above command will return the image id you have to use to run the container you just built:

```console
docker run IMAGE_ID
```

At any point you can list all current container with:

```console
docker ps -a
```

This will start the container, which will run in an endless loop and allow you to execute commands in it using the following command:

```console
docker exec -it CONTAINER_ID COMMAND
```

You can stop the container using:

```console
docker stop CONTAINER_ID
```

And remove it with:

```console
docker remove CONTAINER_ID
```

### Troubleshooting

If you get the message containing permission denied, try to run with sudo or google it. there is much help with this.

### Webserver

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

### Docker API

There is a RESTful API written with the python libary FastAPI for communicating with docker container that can be found in `/docker`

The API needs to run on `127.0.0.1:8000`. To start the API go to `/docker` and run

```bash
uvicorn app:app --reload 
```

Don't use `--reload` when deploying in production
