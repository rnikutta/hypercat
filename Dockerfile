# Multi-stage Dockerfile for Hypercat
#
# Build:
#   docker build -t hypercat .
#
# Run Jupyter (mount local data directory and notebooks):
#   docker run --rm -p 8888:8888 -v /path/to/data:/data -v $(pwd):/work hypercat
#
# Run a one-off Python script:
#   docker run --rm -v /path/to/data:/data hypercat python /work/myscript.py

# ── Stage 1: build environment ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System libraries required by h5py, matplotlib, and scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libhdf5-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the package metadata first so Docker can cache the pip layer
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install ".[dev]"

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Hypercat"
LABEL org.opencontainers.image.description="Hypercubes of AGN Tori"
LABEL org.opencontainers.image.source="https://github.com/rnikutta/hypercat"

# Runtime system libraries only (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-103 \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the build stage
COPY --from=builder /install /usr/local

# /data  — mount your CLUMPY HDF5 data files here
# /work  — mount your notebooks or scripts here
VOLUME ["/data", "/work"]

WORKDIR /work

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''"]
