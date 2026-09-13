FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu
WORKDIR /app
RUN pip install --no-cache-dir runpod requests
COPY src/ /app/src/
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "/app/src/handler.py"]
