# Use an official Python runtime as a parent image
FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgraphviz-dev \
    gcc \
    libc-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY . $APP_HOME

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8080

# Run app.py when the container launches
CMD ["python3", "storage_flask_main.py"]
