# Use the official Ubuntu base image
FROM python:3.12-slim

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpq-dev \
    git \
    curl \
    wget \
    psutils \
    htop \
    nano \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY ../Ginver-main-nist .

# COPY ../Ginver-main-resnet .

# Copy requirements.txt first (to leverage caching)
COPY dockerfile/requirements.txt .

# Install torch directly using pip
RUN pip install -r requirements.txt

# Specify the command to run on container start
CMD ["bash"]
