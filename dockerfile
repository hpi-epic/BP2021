FROM pure/python:3.8-cuda10.2-runtime

WORKDIR /app
EXPOSE 6006

# Do not buffer stdout so we can see it live
ENV PYTHONUNBUFFERED 1

# copy the src folder and install the pip requirements (includes our project as a pip dependency)
RUN python3 -m ensurepip --upgrade
RUN pip install --upgrade pip
COPY requirements.txt .
COPY ./src ./src
RUN python3 -m pip install -r requirements.txt

# copy all relevant files to the container
COPY ./data ./data

# Keep the container running until manually stopped
ENTRYPOINT ["python3", "USER_COMMAND_PLACEHOLDER"]
