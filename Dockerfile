# Use Runpod PyTorch base image
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
# This ensures Python output is immediately visible in logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy App
COPY . /app
RUN uv sync --locked

CMD [ "uv", "run", "handler.py" ]
