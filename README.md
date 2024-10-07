# RMS Norm

This code implements [RMS Norm calculation](https://arxiv.org/pdf/1910.07467). The main implementation is in CUDA C++, with Python wrappers, to expot the CUDA code to python.

## Installation

Please, make sure you have a laptop with NVIDIA toolbox installed and a GPU:
```
python setup.py build
```
If you want to use a `conda` environment you can do the following:
```
conda create -n cuda python=3.11
pip install numpy

export CPATH=/usr/local/cuda/target/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$D_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

python setup.py build
```


### Examples

Once the installation has finished you can use the code as:
```
import numpy as np
from cuda_ops import rms_norm

m = np.random.randn(2,2)
rms_norm(m)
```

# General process to explain what's been built

This has been my stream of work to create the current repo and features. The main requests were:
- build the docker container
- deploy the container
- create a CI/CD
- general fix of the project
and some bonus points as:
- implement a nightly build CI
- enhance the performance for the CUDA C++ code
- redesign the code
