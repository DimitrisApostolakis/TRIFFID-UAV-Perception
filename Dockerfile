# TRIFFID UAV perception - FSS-SAM3 points + polygons.
# Weights (sam3.pt, BPE vocab) and the support bank are mounted at run time, never baked in.
# CUDA 12.8 PyTorch covers Turing through Blackwell (RTX 20xx-50xx, sm_75-sm_120); needs NVIDIA driver >= 570.
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG FSS_SAM3_REPO=https://github.com/WongKinYiu/FSS-SAM3.git
ARG FSS_SAM3_REF=b82ba838678ae8c2f27f618d9500a50c171e223e

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# FSS-SAM3 source, pinned. It is put on sys.path by the backend (no pip install needed).
RUN git clone --filter=blob:none "$FSS_SAM3_REPO" /opt/fss-sam3 \
 && git -C /opt/fss-sam3 checkout "$FSS_SAM3_REF"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps .

# inputs/ input_topview/ outputs/ supports/ live under /app; weights under /models.
RUN mkdir -p inputs input_topview outputs supports /models
ENTRYPOINT ["python", "-m", "triffid_uav"]
CMD ["run", "--config", "configs/default.yaml"]
