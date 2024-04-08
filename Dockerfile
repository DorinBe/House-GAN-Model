# Use an official Python runtime as a parent image
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# # Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
# Define environment variable
# ENV NAME World
# CD app
# Run app.py when the container launches
CMD ["python3", "main.py"]
