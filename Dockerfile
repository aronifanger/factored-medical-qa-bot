# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the project files into the container
COPY . .

# Install project dependencies using uv with proper index strategy
RUN uv pip install --system --index-strategy unsafe-best-match -e .

# The command to run the services will be specified in the docker-compose.yml
# This Dockerfile will be used as a base for both services. 