FROM vllm/vllm-openai:latest

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    AI_FLUX2_PORT=8910

WORKDIR /opt/flux-vast-min-server
COPY requirements.txt /opt/flux-vast-min-server/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY flux_vast_min_server.py /opt/flux-vast-min-server/flux_vast_min_server.py
EXPOSE 8910
CMD ["python3", "/opt/flux-vast-min-server/flux_vast_min_server.py", "--host=0.0.0.0", "--port=8910"]
