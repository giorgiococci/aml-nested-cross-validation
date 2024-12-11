# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the cross-validation-common directory contents into the container at /app
COPY src/cross-validation-common /app/src/cross-validation-common

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the cross-validation-common library directly
RUN pip install /app/src/cross_validation_common