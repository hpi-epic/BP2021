FROM johannschulzetast/miniconda-cuda:11.5-devel-ubuntu20.04

WORKDIR /app
EXPOSE 6006

# Copy yaml for environment creation
COPY environment.yml .

# create conda environment and make the shell use it by default
RUN conda env create -f environment.yml -n dockervenv
ENV PATH /opt/conda/envs/dockervenv/bin:$PATH
# Do not buffer stdout so we can see it live
ENV PYTHONUNBUFFERED 1

# copy the src folder and install the pip requirements (includes our project as a pip dependency)
COPY requirements.txt .
COPY ./src ./src
RUN pip install -r requirements.txt

# copy all relevant files to the container
COPY config.json .
COPY ./results/monitoring ./results/monitoring

# Keep the container running until manually stopped
ENTRYPOINT ["tail", "-f", "/dev/null"]
