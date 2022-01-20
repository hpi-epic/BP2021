FROM gpuci/miniconda-cuda:11.5-devel-ubuntu20.04

WORKDIR /app
EXPOSE 6006
# Create the environment:
COPY ./src ./src
COPY environment.yml .
RUN conda env create -f environment.yml -n dockervenv
SHELL ["conda", "run", "-n", "dockervenv", "/bin/bash", "-c"]
# Activate the environment, and make sure it's activated:
ENV CONDA_DEFAULT_ENV dockervenv
# RUN conda init bash
# RUN source ~/.bashrc
# RUN conda activate dockervenv
RUN echo "Make sure numpy is installed:"
RUN python -c "import numpy"
# copy all relevant files to the container
COPY config.json .
COPY ./results ./results
# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "dockervenv", "python3", "src/monitoring/exampleprinter.py"]