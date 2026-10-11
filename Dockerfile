# Containerised Google Coral Edge TPU YOLO service.
#
# Everything the Coral needs is vendored into ./vendor because Google has
# retired the Coral apt repo (packages.cloud.google.com/apt/...edg-tpu is 404)
# and pycoral is not on PyPI (the "pycoral" package there is an unrelated
# Allen Coral Atlas CLI).
#
# pycoral 2.0.0 ships cp36..cp39 linux wheels only -> Python 3.9 base.
FROM python:3.9-slim-bullseye

# Debian bullseye is EOL: its apt endpoints are gone, so pull from archive.debian.org.
RUN set -eux; \
    printf 'deb http://archive.debian.org/debian bullseye main\n' > /etc/apt/sources.list; \
    printf 'deb http://archive.debian.org/debian-security bullseye-security main\n' >> /etc/apt/sources.list; \
    apt-get -o Acquire::Check-Valid-Until=false update; \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libusb-1.0-0 \
        ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Edge TPU runtime ("direct" = max clock build). pycoral's _pywrap_coral.so
# dlopen()s libedgetpu.so.1.0 at runtime.
COPY vendor/libedgetpu.so.1.0 /usr/local/lib/libedgetpu.so.1.0
RUN ln -sf /usr/local/lib/libedgetpu.so.1.0 /usr/local/lib/libedgetpu.so.1 && ldconfig

COPY vendor/*.whl /tmp/vendor/
RUN pip install --no-cache-dir /tmp/vendor/tflite_runtime*.whl /tmp/vendor/pycoral*.whl \
    && pip install --no-cache-dir \
        "numpy<2" \
        "opencv-python-headless<5" \
        pyyaml \
        pillow \
        fastapi \
        "python-multipart" \
        "uvicorn[standard]" \
    && rm -rf /tmp/vendor

COPY app/ /app/
COPY vendor/models/ /app/models/

WORKDIR /app
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "server.py"]
