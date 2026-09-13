FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu

WORKDIR /app

# Qwen3 inference + document parsers + RunPod
RUN pip install --no-cache-dir \
    runpod \
    requests \
    python-docx \
    transformers \
    accelerate \
    bitsandbytes

COPY src/ /app/src/
COPY tests/ /app/tests/

ENV PYTHONUNBUFFERED=1
ENV QWEN_MODEL=Qwen/Qwen3-8B
ENV QWEN_MAX_NEW_TOKENS=4096
ENV QWEN_MAX_INPUT_CHARS=100000
ENV QWEN_LOAD_IN_4BIT=true

CMD ["python", "-u", "/app/src/handler.py"]
