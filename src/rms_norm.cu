#include <cmath>
#include <Python.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


// Define a macro for checking CUDA errors
#define cudaCheckError(call) {                                          \
    cudaError_t e=call;                                \
    if(e!=cudaSuccess) {                                             \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(0);                                                    \
    }                                                               \
}


__global__ void rmsNormalizationKernel(double *matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            sum += matrix[row * cols + i] * matrix[row * cols + i];
        }
        double rms = sqrt(sum / cols);
        for (int i = 0; i < cols; ++i) {
            matrix[row * cols + i] /= rms;
        }
    }
}


static PyObject* rms_norm(PyObject* self, PyObject* args) {
    PyArrayObject *input_matrix;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_matrix)) {
        return NULL;
    }

    double *matrix = static_cast<double*>(PyArray_DATA(input_matrix));
    int rows = PyArray_DIM(input_matrix, 0);
    int cols = PyArray_DIM(input_matrix, 1);

    // Allocate GPU memory and copy data
    double *d_matrix;
    cudaCheckError(cudaMalloc(&d_matrix, rows * cols * sizeof(double)));
    cudaCheckError(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(double), cudaMemcpyHostToDevice));


    // Launch the kernel
    dim3 blockSize(256);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x);
    rmsNormalizationKernel<<<gridSize, blockSize>>>(d_matrix, rows, cols);

    // Copy result back to host
    cudaMemcpy(matrix, d_matrix, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"rms_norm", rms_norm, METH_VARARGS, "Row-wise RMS normalization of a matrix"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "rms_norm",
    "Row-wise RMS normalization of a matrix",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_rms_norm(void) {
    import_array();
    return PyModule_Create(&module);
}
