FROM pure/python:3.8-cuda10.2-runtime

WORKDIR /app
EXPOSE 6006

# Do not buffer stdout so we can see it live
ENV PYTHONUNBUFFERED 1

# copy the src folder and install the pip requirements (includes our project as a pip dependency)
COPY requirements.txt .
COPY ./src ./src
RUN python3 -m ensurepip --upgrad
RUN pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# copy all relevant files to the container
COPY config.json .
COPY ./results/monitoring ./results/monitoring

# Keep the container running until manually stopped
ENTRYPOINT ["tail", "-f", "/dev/null"]
