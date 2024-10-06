# syntax=docker/dockerfile:1-labs
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# install non-Python dependencies
RUN <<EOF
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles
apt-get update
apt-get install -y python-is-python3 python3-pip git
EOF

RUN pip install --upgrade pip setuptools
# build and install the package
WORKDIR /opt/cuda_ops
COPY . .

# add requirements
RUN <<EOF bash -ex
pip install --upgrade pip setuptools setuptools_scm wheel numpy
python setup.py build
pip install -e .[test]
EOF

# run basic tests
RUN <<EOF bash -ex
ls -l /opt/cuda_ops/*
pytest /opt/cuda_ops/test/test.py
EOF
