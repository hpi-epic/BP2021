FROM nvidia/cuda:11.3.0-base-ubuntu20.04
RUN sudo apt-get install -y python 3.8

WORKDIR /app
EXPOSE 6006

# Do not buffer stdout so we can see it live
ENV PYTHONUNBUFFERED 1

# Update pip
RUN python3 -m ensurepip --upgrade
RUN pip install --upgrade pip

# This directory is needed for pip installation
RUN mkdir -p ./recommerce/configuration
# Copy files needed for pip package
COPY README.md README.md
COPY LICENSE LICENSE
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY setup.cfg setup.cfg
# Install the recommerce package
# How can we do this before actually having the code?
# Because we install using -e a symbolic link is created, so any files we copy over into the ./recommerce
# path later are automatically recognised as part of the package
RUN pip install -e .

# ...so now we can copy over the often-changed files *after* installing the dependencies, saving lots of time due to caching!
COPY ./recommerce ./recommerce

# set the datapath and unpack the default data
# we only want the modelfiles and remove the config files to make sure we can only use the ones provided by the user
# (i.e. if the upload fails, the program can't start and won't just use the default one from the recommerce package)
RUN recommerce --datapath . --get-defaults-unpack
RUN rm hyperparameter_config.json
RUN rm environment_config_training.json
RUN rm environment_config_exampleprinter.json
RUN rm environment_config_agent_monitoring.json

# Perform the specified action when starting the container
ENTRYPOINT ["echo", "ENTRYPOINT not overwritten! The container does nothing and will be stopped now. Make sure to start the container using the API, not directly through Docker"]
