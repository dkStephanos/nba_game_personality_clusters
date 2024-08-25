# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY devops/requirements.txt .

# Add the unstable repository and update the package lists
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    echo "deb http://deb.debian.org/debian unstable main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y openjdk-11-jdk wget && \
    wget -qO- https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz | tar xvz -C /usr/local/ && \
    mv /usr/local/spark-3.3.0-bin-hadoop3 /usr/local/spark && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Set JAVA_OPTS to increase the heap space
ENV JAVA_OPTS="-Xmx8g"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install JupyterLab
RUN pip install jupyterlab

# Expose the port for JupyterLab
EXPOSE 8888

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:/app/scripts

# Command to start JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.notebook_dir=/app/notebooks"]
