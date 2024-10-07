# syntax=docker/dockerfile:1-labs
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

WORKDIR /opt/cuda_ops

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip setuptools setuptools_scm wheel numpy
RUN python setup.py build

RUN apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip
