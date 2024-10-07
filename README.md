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
export PYTHONPATH=$(pwd):$PYTHONPATH

python setup.py build
```


## Development best practices

For the development of the code, I suggest you to create a virutal environment, like the `conda` env above, and install pre-commit:
```
pip install pre-commit
pre-commit install
```
This will automatically run sanity checks before pushing the code to GitHub


### Examples

Once the installation has finished you can use the code as:
```
import numpy as np
from cuda_ops import rms_norm_execute

m = np.random.randn(2,2)
rms_norm_execute.compute(m)
```

Pulling and testing Docker image from [DockerHub](https://hub.docker.com/repository/docker/sbsynth/cuda_ops/general):
```
docker pull sbsynth/cuda_ops:0.0.3
docker run -it sbsynth/cuda_ops:0.0.3 bash
```

# General process to explain what's been built

### Refactoring of the Python code

- Refactor the `setup.py` to define the package build and to simplify the installation process. At first, the installation process was handled by a `build.sh` and a `Makefile`. The `build.sh` script was triggered also via `setup.py`. The current setup was using `shutil` and command line argument to run all the build, and copy the compiled CUDA shared library across folders. Although this may be a good initial setup, it is not a great long term solution.
The current solution makes use of a custom build to execute and compile CUDA files through python directly. Moreover, it exploits python build capacities to correctly place shared library files in the Python path.

- Associate a `pyproject.toml` file with `setup.py`. The idea is to have an easy way to keep track of the needed packages for the code, and a dynamic method to update the package version, without human intervention. For doing this, in the `setup.py` I've added `setuptool_scm`, and tailor the script to work with `setuptool_scm`, so that it can read a version during the CI process. The `pyproject.toml` file enhances this behaviour with `dynamic = ["version"]`. The version will then be based on the tags and versions we're creating in the CI process. We will have a specific version for `dev` packages, so `DEV_VERSION` number, when things are pushed to a PR, and a specific version for a stable release, where `STABLE_RELEASE` is the GitHub release version, when a PR gets merged on `main`. Moreover, `pyproject.toml` makes the project compliant with [PEP517](https://peps.python.org/pep-0517/) and [PEP518](https://peps.python.org/pep-0518/), it simplifies the dependency management and it improves the interoperability, as it can work with `pip` and `poetry` and `uv` or other package managers.

- Extension of tests. The initial test checked whether the result was matching CPU expected results. Other starting point tests, that I've introduced, cover the following:
    - Test with simple inputs, that we can compute by hand;
    - Test with a matrix of zeros. This suggests an improvement in the CUDA code or in the Python wrapper, on how to deal with 0s;
    - Test with large values, for example `1e20`;
    - Test with small values, for example `1e-20`. These last two tests suggests we can introduce a type handling in the CUDA code;
    - Test with a matrix whose values are ranging from `1e-20` to `1e20`;
    - Test with a 1D array. This suggests to introduce a check of the input array size in the code.


- Extend the Python code with a better funciton handler. I've created `rms_norm_execute.py` that treats the input NumPy array:
    - check whether the input is a NumPy array;
    - check whether the input is a > 2D array;
    - check if the type is a `np.float64` or `np.float32`;
    - calls the `rms_norm` function.
Further work can be done on this side, by, for example, hiding the real `rms_norm` function to public.


### Create a CI/CD

For the CI/CD I used my personal GitHub, running CI/CD on GitHub hosted runners. An important caveat is that, at the moment, we do not have runners with GPU available to individuals, therefore the process does build the package, but it cannot execute the tests, as a GPU is missing.
However, I've implemented the main logic to allow to have the build, the test (this step is commented out) and the deployment of the package. The same applies for publishing the package to Pypi.org.
The philosophy of the current setup follows 2 simple points:
1. We have a CI/CD that automatically handles the versioning and the "night" versions
2. We have a CI/CD that is modular enough to accommodate a distribution across different Cloud artifact registry (e.g. ECR, GCR and so on )

For implementing the CI I've run the following actions:
- implement pre-commits to verify input files, perform ruff and linting and general cleaning. The pre-commit allow a smoother interaction with public and developers. The introduced checks correct trailing the general code writing (e.g. trailing white spaces, end of line in a file ), avoid people to push directly on the `main` branch, run `ruff` and it automatically updates the `requirements.txt` file for any new package introduced in the `pyproject.toml` file.
- implement the CI as GitHub Actions. All the actions can be inspected in `.github/workflows`. In particular:
    - we have `on_pr.yml` action that kicks in when there's a new PR;
    - `on_push_main.yml` when a PR is merge (commit-merge) to main;
    - `nightly_build.yml` that works everyday at midnight UK time, to create a nightly build version of the package.
- Each action has the following logic:
    - Execute `push_python_package.yml`. This computes a new version of the package. Then, it can build the package wheel and prepare everything for Pypi.org
    - The version is computed through `.github/workflows/compute_semantic_version/action.yml`. [Semantic versioning](https://semver.org/) follows the logic to versioning based on the commit message, to detect MAJOR, MINOR or PATCH changes. The commit message can have then a define structure, so that with `Breaking changes` we're triggering a Major version release, `fix` a Minor change, `feature` a Patch change
    - `build_and_push_image.yml`: this action creates the package container with docker, and push it to [DockerHub](https://hub.docker.com/r/sbsynth/cuda_ops/tags).
Working examples of the CI can be viewed on my [GitHub page](https://github.com/Steboss/cuda_ops/actions).

Future work should involve the usage of GPU based machines, as well as running on different Python versions and different machines (e.g. windows, macos and linux).

### Optimisation of the CUDA C++ Code

The improvements that I brought in the CUDA space are mainly:
- Error handling: I add a better error handling mechanism, defining a custom macro to catch and report CUDA-specific error, ensuring a smoother debugging experience and code stability;
- Added a support for multiple precisions: I refactored the code, so taht is now possible to deal with `float32` and `float64` matrices inputs, providing more flexibility and better support for end users (i.e. they do not have to convert arrays to other precision formats);
- Optimised the memory management with CUDA streams: the usage of CUDA streams enables asynchronous memory operations, reducing a possible bottlenecks at the CPU level. Indeed, it should prevent `cudaMemcpy` from blocking the CPU, that may be beneficial for large matrices. Like wise, the same apply for the data transfer, so that the CPU can execute other downstream tasks without waiting for the entire memory transfer to complete.

Future optimisation work should be done on the shared memory front, to further optimize how the matrices are handled on the GPU itself.
