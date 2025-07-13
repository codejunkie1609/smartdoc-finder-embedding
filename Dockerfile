# ./smartdoc-finder-embedding/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# ✅ STEP 1: Install system-level build tools, including gcc
# This will allow pip to compile packages with C extensions.
# The --no-install-recommends flag keeps the image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Set the working directory in the container
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
# ✅ STEP 2: Now this pip install command will succeed
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# NOTE: No ENTRYPOINT or CMD is needed here.
# The 'command' in docker-compose.yml will provide the command to run.
