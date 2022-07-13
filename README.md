# Online Marketplace Simulation: A Testbed for Self-Learning Agents  <!-- omit in toc -->

![CI](https://github.com/hpi-epic/BP2021/actions/workflows/CI.yml/badge.svg)
![Coverage-Badge](/badges/coverage.svg)
![Docstring-Coverage](/badges/docstring_coverage.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

- [1. The `recommerce` package](#1-the-recommerce-package)
	- [1.1. Quick Start Guide](#11-quick-start-guide)
- [2. Installing dependencies](#2-installing-dependencies)
	- [2.1. Dependency Installation Troubleshooting](#21-dependency-installation-troubleshooting)
- [3. Installing the `Recommerce` package](#3-installing-the-recommerce-package)
- [4. Testing](#4-testing)
	- [4.1. Pytest](#41-pytest)
		- [4.1.2. Pytest Plugins](#412-pytest-plugins)
	- [4.2. Webserver Tests](#42-webserver-tests)
	- [4.3. Coverage](#43-coverage)
- [5. Pre-commit](#5-pre-commit)
	- [5.1. Interrogate](#51-interrogate)
	- [5.2. Pre-commit Troubleshooting](#52-pre-commit-troubleshooting)
- [6. Networking Scenario](#6-networking-scenario)
	- [6.1. Docker](#61-docker)
		- [6.1.1. Docker-API](#611-docker-api)
		- [6.1.2. Using Docker natively](#612-using-docker-natively)
		- [6.1.3. Docker/GPU-VM Troubleshooting](#613-dockergpu-vm-troubleshooting)
	- [6.2. Webserver](#62-webserver)
	- [6.3. Docker API](#63-docker-api)
- [7. TensorBoard](#7-tensorboard)

Working repository in context of the bachelor's project "*Online Marketplace Simulation: A Testbed for Self-Learning Agents*" at the *Enterprise Platform and Integration Concepts* ([@hpi-epic](https://github.com/hpi-epic)) research group of the Hasso-Plattner-Institute.

*TODO: Project summary*

## 1. The `recommerce` package

If you have not yet done so, first install the `recommerce` package by following [Installing dependencies](#2-installing-dependencies) and [Installing the `recommerce` package](#3-installing-the-recommerce-package).

At any point after the installation has completed, use

```terminal
recommerce --help
```

in your terminal to see the usage options of the package. The Quick Start Guide in the next section will help you set up the simulation framework and get started with a first experiment.

### 1.1. Quick Start Guide

The `recommerce` package requires users to provide it with a datapath, which is where the package will look for configuration files and write output files, such as statistics and diagrams.
During installation of the package, the datapath was set to the current working directory. If you want to modify the datapath, you can use the following command:

```terminal
recommerce --datapath "your_preferred_path"
```

You should see the following message indicating your path is valid:

```terminal
Data will be read from and saved to "your_preferred_path"
```

You can check the currently set datapath at any point by running

```terminal
recommerce --get-datapath
```

To start an experiment, `recommerce` requires you to provide configuration files, which contain the necessary information to set up the simulation. You can either write those files yourself or use the following command to have `recommerce` copy over default files which you can immediately use or modify as you wish:

*WARNING*: By using the following command, any files with the same names as the default data will be overwritten, so use with caution! To get the data in a folder called `default_data` use the flag `--get-defaults` instead.

```terminal
recommerce --get-defaults-unpack
```

The `-unpack` part of the flag makes sure that the default files are not just stored in the `default-data` folder in your datapath, but unpacked in a way that `recommerce` can immediately find them, by storing configuration files and pre-trained models in the `configuration_files` and `data` directories respectively.

Now you are ready to run your first exampleprinter session:

```terminal
recommerce -c exampleprinter
```

If you are using the default configuration files provided by `recommerce --get-defaults`, this exampleprinter run will also create an in-depth HTML-Slideshow of the run in the `results/exampleprinter` folder within your datapath.

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

If version numbers have changed in the `environment.yml` it can happen that conda finds conflicts and tries resolving them without succeeding. In this case you may need to reinstall the packages. Deactivate your environment before proceeding, otherwise conda cannot perform the commands:

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

Before proceeding, please inform yourself on whether or not your device supports `cuda`, take a look at [this](https://developer.nvidia.com/cuda-gpus) resource provided by NVIDIA. This decides if you should install our project with cuda support, or without, which comes down to the specific version of `torch` that will be installed. The file size of the torch version that supports cuda is much larger in comparison. 

If your device supports cuda and you want to utilize its capabilities, use the following command within the project directory:

```terminal
pip install -e .[gpu] -f https://download.pytorch.org/whl/torch_stable.html
```

Otherwise, to install without cuda support, use:

```terminal
pip install -e .[cpu]
```

This installs the `recommerce` folder (and its subdirectories) as a local pip package. The `-e` flag indicates to pip that the package should be installed in an editable state. This results in the packages not being directly written to where pip dependencies usually would, but only a "link" to you current working directory being created. In order to install the package, we use `setuptools`, which uses the `setup.py`, `setup.cfg` and `pyproject.toml` files located in the root directory of the project. The `setup.cfg` file includes all necessary metadata needed to correctly install the project.

If you want to properly install the project as a pip package, omit the `-e` flag. This will "copy" the packages into your pip-installation folder (when using conda, this will be `Path/To/anaconda3/envs/your_venv_name/Lib/site-packages`), meaning that any changes to your source-code will only be reflected when installing the package again, therefore you should not use that command if you plan on changing the code.

You can confirm that the installation was successful by checking

```terminal
recommerce --version
```

## 4. Testing

### 4.1. Pytest

[Pytest documentation](https://docs.pytest.org/en/latest/index.html)

If you want to run tests locally you can do so in a few ways:

Simply running all tests:

```terminal
pytest
```

Running all tests with increased verbosity:

```terminal
pytest -v
```

#### 4.1.2. Pytest Plugins

__Markers__

We have also added markers to some tests, such as `slow` or `training`. You can filter tests using the `-m` flag, if for example
you want to exclude `slow` tests from the run, use:

```terminal
pytest -m "not slow"
```

__Pytest-randomly__

To make sure that our tests do not have hidden dependencies between each other, their order is shuffled every time the suite is run.
For this, we use the [pytest-randomly](https://pypi.org/project/pytest-randomly/) plugin. Before test collection, a `--randomly-seed` is set and printed to the terminal, which you can use in sbsequent runs to repeat the same order of tests, which can be useful for debugging. 

__Pytest-xdist__

[pytest-xdist](https://pypi.org/project/pytest-xdist/) is a plugin which can be used to distribute tests across multiple CPUs to speed up test execution. See their documentation for usage info.

### 4.2. Webserver Tests

To run tests you have written for the Django webserver go into the `webserver` folder and run

```terminal
python ./manage.py test -v 2
```

### 4.3. Coverage

[Coverage.py Documentation](https://coverage.readthedocs.io/en/6.1.2/)

If you want to track the current test coverage use these commands:

- Run all tests, collect coverage info, write it to the coverage.json (for source code analysis of coverage) and update the `coverage.svg` badge:

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

We are using `pre-commit` to lint and format our files before committing. Pre-commit itself should already have been installed through the `environment.yml`. Initialize pre-commit before the first commit using

```terminal
pre-commit install
```

To circumvent possible errors caused later on, we recommend also running pre-commit once using the following command:

```terminal
pre-commit run --all-files
```

which will install the needed environment.

From now on, pre-commit will always be run automatically before committing. It is also run as part of our CI pipeline.

### 5.1. Interrogate

[Interrogate documentation](https://interrogate.readthedocs.io/en/latest/)

We use Interrogate to monitor our docstring coverage. The plugin is configured within the `pyproject.toml` in the root directory of the project. Interrogate is automatically run with Pre-commit, but the badge can only be updated manually.

To update the badge, modify the `pre-commit-config.yml` file by swapping the line setting the `args` for the plugin and then running pre-commit locally. Do not commit this modified `pre-commit-config.yml` file to Github, as this will break the CI pipeline.

### 5.2. Pre-commit Troubleshooting

If you get the following error:

```text
Git: Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases
```

while trying to commit, the cause is most likely `pre-commit` trying to access a Python version not in your virtual environment.

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

To use Docker, first install it on your machine, for more convenience we recommend using [Docker Desktop](https://www.docker.com/products/docker-desktop/). Afterwards, you can build the image used in our repository using the following command:

```terminal
python ./docker/docker_manager.py
```

#### 6.1.1. Docker-API

We recommend interacting with Docker using the Webserver as outlined in the [Webserver](#62-webserver) section. If you still want to use Docker from you commandline, refer to [Using Docker natively](#612-using-docker-natively).

#### 6.1.2. Using Docker natively

You can also build the recommerce image using the following command while in the root directory containing the `dockerfile`:

```terminal
docker build . -t recommerce
```

Building the image may take a while, it is about 7GB in size. To see all current images on your system use:

```terminal
docker images
```

You can create and run a container for the recommerce image using the following command:
Note that if your machine does not have a dedicated GPU, you may need to omit the `--gpus all` flag.

```terminal
docker run -it --entrypoint /bin/bash --gpus all recommerce
```

Running this command will start a container and automatically open an interactive shell for you. Here you can now perform any command you like, start by using:

```terminal
recommerce --help
```

At any point you can list all current containers with:

```terminal
docker ps -a
```

You can stop a container using its ID:

```terminal
docker stop CONTAINER_ID
```

And remove it with:

```terminal
docker rm CONTAINER_ID
```

#### 6.1.3. Docker/GPU-VM Troubleshooting

This section is aimed at developers, and any errors described here should only occur when trying to deploy the webserver/docker to a new environment/virtual machine.

##### 6.1.3.1. Running a docker container with GPU-support <!-- omit in toc -->

When trying to run a docker container (with a gpu device request), you get the following error:

```text
failed to create shim: OCI runtime create failed: container_linux.go:380: starting container process caused: process_linux.go:545: container init caused: Running hook #0:: error running hook: signal: segmentation fault, stdout: , stderr:: unknown
```

This error is caused by your local linux distribution (on Windows this pertains to the WSL instance used by docker) not having required packages installed needed to support cuda.
A proposed workaround is to update/downgrade the following packages:

```terminal
apt install libnvidia-container1=1.4.0-1 libnvidia-container-tools=1.4.0-1 nvidia-container-toolkit=1.5.1-1
```

Issues in the `nvidia-docker`-repository that describe this error can be found [here](https://github.com/NVIDIA/nvidia-docker/issues/1533), [here](https://github.com/NVIDIA/nvidia-docker/issues/1534), and [here](https://github.com/NVIDIA/nvidia-docker/issues/1536). Please note that we have not confirmed that the workaround solves this problem.


##### 6.1.3.2. Starting a training session with GPU-support <!-- omit in toc -->

*Note: This error should no longer occur if the recommerce package was installed with the correct extra selected. We are still including this section for completeness.*

When trying to start a training session on the VM or your machine (e.g. using `recommerce -c training`) you get the following error:

```terminal
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, apb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```

This error comes from your torch installation not having cuda-support, but your machine supporting cuda. You should confirm that you installed the correct version of recommerce in the [Installing the `Recommerce` package](#3-installing-the-recommerce-package) section. In this case, you should install recommerce with the `gpu` extra, which installs the following versions of torch:

```text
torch==1.11.0+cu115
torchvision==0.12.0+cu115
torchaudio==0.11.0+cu115
```

You can also manually update these versions using

```terminal
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
```

### 6.2. Webserver

We provide a Django Webserver with a simple user interface to manage the docker containers.
To use the webserver you need to either have a `.env.txt` in the `BP2021/webserver` directory or have the environment variables `SECRET_KEY` and `API_TOKEN` set.

Here is an example for a `.env.txt`

```text
this_line_contains_the_secret_key_for_the_django_server
this_line_contains_the_master_secret_for_the_api
```

Remember to change these secrets when they are leaked to the public. Both secrets should be random long strings. Keep in mind, that the master_secret for the API (`API_TOKEN`) should be equal to the `AUTHORIZATION_TOKEN` on the API side.

When starting the webserver, you will notice the login page. To create a *superuser* and login to the page, change to the webserver directory and run:

```terminal
python ./manage.py createsuperuser
```

To manage your other users, go to `127.0.0.1:2709/admin` and login with the credentials you provided when creating the superuser.

To start the webserver on `127.0.0.1:2709` go to `/webserver` and start the server by using the following command

```terminal
python ./manage.py runserver 2709
```

When you add more fields to the database model or you change existing fields, you need to run

```terminal
python ./manage.py makemigrations
```

Before starting the server you might need to apply any pending migrations using

```terminal
python ./manage.py migrate
```

To run tests you have written for the Django webserver go into the *webserver* folder and run

```terminal
python ./manage.py test -v 2
```

### 6.3. Docker API

There is a RESTful API written with the python library FastAPI for communicating with docker containers that can be found in the `/docker` directory.

The API needs to run on `127.0.0.1:8000`. To start the API go to the `/docker` directory and run

```terminal
uvicorn app:app --reload
```

Don't use `--reload` when deploying in production.

Alternatively, you can also run the `app.py` from the docker directory as well.

If you want to use the API, you need to provide an `AUTHORIZATION_TOKEN` in your environment variables. For each API request the value at the authorization header will be checked. You can only perform actions on the API when this value is the same as the value in your environment variable.

WARNING: Please keep in mind that the `AUTHORIZATION_TOKEN` must be kept a secret. If it is revealed you need to revoke it and set a new secret. Furthermore, think about using transport encryption to ensure that the token won't get stolen on the way.

## 7. TensorBoard

TensorBoard is used extensively to track parameters of the training as well as the agents and the market itself. It is started from the console by using:

```terminal
tensorboard serve --logdir your_log_path
```

The path specified can be changed to just include one subfolder of the runs folder to track just one of the experiments.
NOTE: TensorBoard might not work with Safari, but Chrome does the job.
