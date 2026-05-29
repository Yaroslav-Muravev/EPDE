# python:3.10-slim has glibc, just enough to install the wheels.
# numpy / scipy / sklearn / torch all ship manylinux wheels, so we
# don't need a build toolchain at runtime.
FROM python:3.10-slim

# libgomp1 is needed by numpy/sklearn/torch for OpenMP threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install Python deps first so the layer caches across source edits.
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the repo last so source edits don't bust the dep cache.
COPY . /work

# Default to interactive shell; the actual sweep command is supplied
# by docker-compose via the per-service ``command:`` field.
ENTRYPOINT ["/bin/bash", "-l"]
