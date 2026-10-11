#!/usr/bin/env bash
# Build + push the Coral images on the x86 Docker daemon (the Coral node itself
# has no Docker, and pycoral wheels are x86_64-only).
set -euo pipefail

DOCKER_HOST="${DOCKER_HOST:-tcp://192.168.11.128:2375}"
export DOCKER_HOST
TAG="${TAG:-dev}"

cd "$(dirname "$0")"

echo "==> building initialed85/coral-yolo:${TAG}"
docker build -t "initialed85/coral-yolo:${TAG}" .
docker push "initialed85/coral-yolo:${TAG}"

echo "==> building initialed85/coral-plugin:${TAG}"
docker build -f Dockerfile.plugin -t "initialed85/coral-plugin:${TAG}" .
docker push "initialed85/coral-plugin:${TAG}"
