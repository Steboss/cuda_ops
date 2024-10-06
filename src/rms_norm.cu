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

__global__ void rmsNormalizationKernel(float *matrix, int rows, int cols) {
    // shard memory for row elements
    extern __shared__ float rowElements[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0;
        for (int i = threadIdx.y; i < cols; i+= blockDim.y) {
            rowElements[i] = matrix[row*cols+i];
            sum += rowElements[i] * rowElements[i];
        }
        float rms = sqrt(sum / cols);
        for (int i = threadIdx.y; i< cols; i += blockDim.y) {
            matrix[row * cols + i] = rowElements[i]/rms;
        }
    }
}


static PyObject* rms_norm(PyObject* self, PyObject* args) {
    PyArrayObject *input_matrix;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_matrix)) {
        return NULL;
    }

    float *matrix = static_cast<float*>(PyArray_DATA(input_matrix));
    int rows = PyArray_DIM(input_matrix, 0);
    int cols = PyArray_DIM(input_matrix, 1);

    // Allocate GPU memory and copy data
    float *d_matrix;
    cudaCheckError(cudaMalloc(&d_matrix, rows * cols * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    int minGridSize, blockSize;
    cudaCheckError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rmsNormalizationKernel, 0, rows * cols));
    int gridSize = (rows + blockSize -1)/blockSize;
    std::cout << "Optimal block size: " << blockSize << ", Grid size: " << gridSize << std::endl;

    rmsNormalizationKernel<<<gridSize, blockSize>>>(d_matrix, rows, cols);
    cudaCheckError(cudaGetLastError());
    // Copy result back to host
    cudaCheckError(cudaMemcpy(matrix, d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_matrix));
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
