# Use the official Ubuntu base image
FROM python:3.12-slim

COPY ../Ginver-main .

# Copy requirements.txt first (to leverage caching)
COPY dockerfile/requirements.txt .

# Install torch directly using pip
RUN pip install -r requirements.txt

# Specify the command to run on container start
CMD ["bash"]
