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
RUN recommerce --datapath . --get-defaults-unpack

# Perform the specified action when starting the container
ENTRYPOINT ["echo", "ENTRYPOINT not overwritten! The container does nothing and will be stopped now. Make sure to start the container using the API, not through directly through Docker"]

