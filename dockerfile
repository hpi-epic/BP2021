# On your first run, you need to use the following `FROM` command instead of `FROM bp2021default`:
# FROM gpuci/miniconda-cuda:11.5-devel-ubuntu20.04
# also, you need to update the conda environment creation (line 17) to `create` instead of `update`!
# and run the following command to create the default docker image:
# docker build . -t bp2021default
# afterwards, bp2021default will be used as the template, where just the changed files are updated
# watch out, each new image and container will need to be deleted after running, or you will clutter your harddrive!
FROM bp2021default

WORKDIR /app
EXPOSE 6006

# Copy yaml for environment creation
COPY environment.yml .

# create conda environment and make the shell use it by default
RUN conda env update -f environment.yml -n dockervenv
SHELL ["conda", "run", "-n", "dockervenv", "/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV dockervenv

# copy the src folder and install the pip requirements (includes our project as a pip dependency)
COPY requirements.txt .
COPY ./src ./src
RUN pip install -r requirements.txt

# copy all relevant files to the container
COPY config.json .
COPY ./results ./results

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "dockervenv", "python3", "src/test.py"]
