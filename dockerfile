FROM pure/python:3.8-cuda10.2-runtime

WORKDIR /app
EXPOSE 6006

# Do not buffer stdout so we can see it live
ENV PYTHONUNBUFFERED 1

# Update pip
RUN python3 -m ensurepip --upgrade
RUN pip install --upgrade pip

# Copy files needed for pip package
COPY ./recommerce ./recommerce
COPY pyproject.toml pyproject.toml
COPY setup.cfg setup.cfg
COPY setup.py setup.py
COPY LICENSE LICENSE
COPY README.md README.md
# Install the recommerce package
RUN pip install .
# set the datapath and unpack the default data
# we only want the modelfiles and remove the config files to make sure we can only use the ones provided by the user
# (i.e. if the upload fails, the program can't start and won't just use the default one from the recommerce package)
RUN recommerce --datapath . --get-defaults-unpack
RUN rm hyperparameter_config.json
RUN rm environment_config_training.json
RUN rm environment_config_exampleprinter.json
RUN rm environment_config_agent_monitoring.json

# Perform the specified action when starting the container
ENTRYPOINT ["echo", "ENTRYPOINT not overwritten!"]
