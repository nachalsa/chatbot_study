# syntax=docker/dockerfile:1
FROM python:3.10

# install app dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Copy the entire project into the working directory
# COPY . /app
COPY .env.* ./
# final configuration
# ENV FLASK_APP=hello
EXPOSE 8000
CMD ["fastapi", "run", "main.py"]