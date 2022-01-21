FROM gpuci/miniconda-cuda:11.5-devel-ubuntu20.04

WORKDIR /app
EXPOSE 6006

# Copy yaml for environment creation
COPY environment.yml .

# create conda environment and make the shell use it by default
RUN conda env create -f environment.yml -n dockervenv
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
ENTRYPOINT ["conda", "run", "-n", "dockervenv", "python3", "src/monitoring/exampleprinter.py"]