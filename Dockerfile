# syntax=docker/dockerfile:1-labs
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# install non-Python dependencies
RUN <<EOF
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles
apt-get update
apt-get install -y python-is-python3 python3-pip
EOF

# build and install the package
WORKDIR /opt/cuda_ops
ADD cuda_ops $WORKDIR/cuda_ops
ADD src $WORKDIR/src
ADD test $WORKDIR/test
ADD pyproject.toml $WORKDIR/pyproject.toml
ADD setup.py $WORKDIR/setup.py

RUN <<EOF bash -ex
pushd /opt/cuda_ops
pip install -e .[test]
python setup.py build
EOF

# run basic tests
RUN <<EOF bash -ex
pytest /opt/cuda_ops/test/test.py
EOF
