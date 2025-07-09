# In ./smartdoc-finder-embedding/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# NOTE: No ENTRYPOINT or CMD is needed here.
# The 'command' in docker-compose.yml will override it.