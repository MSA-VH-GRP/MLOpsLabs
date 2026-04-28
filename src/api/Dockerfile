FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 via deadsnakes PPA (Ubuntu 22.04 default is 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gpg-agent \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    gcc g++ make librdkafka-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip     pip     /usr/local/bin/pip3.11 1

# CUDA paths for nvcc
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

COPY pyproject.toml .

# PyTorch cu121 — pinned to match CUDA 12.1 base image
RUN pip install --no-cache-dir "torch==2.5.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# Only compile for SM 8.6 (RTX 3080 Laptop) → 3-5 min instead of 30 min.
# MAX_JOBS=2 limits parallel nvcc workers to prevent Docker Desktop OOM.
# --no-build-isolation reuses the pinned torch==2.5.1+cu121 in the build env.
# CFLAGS/CXXFLAGS: force pre-CXX11 ABI to match PyTorch whl (compiled_with_cxx11_abi=False).
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV MAX_JOBS=2
ENV CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

RUN pip install --no-cache-dir --no-build-isolation "causal-conv1d>=1.2.0"
RUN pip install --no-cache-dir --no-build-isolation mamba-ssm

# Reset env vars (not needed at runtime)
ENV TORCH_CUDA_ARCH_LIST=""
ENV MAX_JOBS=""
ENV CFLAGS=""
ENV CXXFLAGS=""

# Install project dependencies
# --ignore-installed: skip uninstalling distutils-managed packages that pip
# cannot remove (e.g. setuptools installed by deadsnakes' Python 3.11 package)
RUN pip install --no-cache-dir --ignore-installed -e .

# Re-pin torch to cu121 after project deps may have upgraded it to a newer CUDA build.
# mamba-ssm was compiled against torch==2.5.1+cu121; the ABI must match at runtime.
RUN pip install --no-cache-dir --force-reinstall "torch==2.5.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

COPY src/ ./src/
COPY feast/ ./feast/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
