FROM nvidia/cuda:11.3.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractiv
# Install Python
RUN apt update && apt install -y --no-install-recommends python3 python3-pip \
	&& ln -sf python3 /usr/bin/python \
	&& ln -sf pip3 /usr/bin/pip \
	&& pip install --upgrade pip \
	&& pip install wheel setuptools docker

WORKDIR /app
EXPOSE 6006

# Do not buffer stdout so we can see it live
ENV PYTHONUNBUFFERED 1

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
RUN pip install -e .[gpu] -f https://download.pytorch.org/whl/torch_stable.html

# ...so now we can copy over the often-changed files *after* installing the dependencies, saving lots of time due to caching!
COPY ./recommerce ./recommerce

# Install again but without the -e flag to fix a weird bug where pip doesn't recognize the package otherwise
# we still want to install with -e earlier to be able to use caching for the large dependencies!
RUN pip install .[gpu] -f https://download.pytorch.org/whl/torch_stable.html

# set the datapath and unpack the default data
# we only want the modelfiles and remove the config files to make sure we can only use the ones provided by the user
# (i.e. if the upload fails, the program can't start and won't just use the default one from the recommerce package)
RUN recommerce --datapath . --get-defaults-unpack
RUN rm -r configuration_files

# This is a placeholder Entrypoint that will be overwritten when creating a container with a specified task
ENTRYPOINT ["echo", "ENTRYPOINT not overwritten! The container does nothing and will be stopped now. Make sure to start the container using the API, not directly through Docker."]
