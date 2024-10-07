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


template <typename T>
__global__ void rmsNormalizationKernel(T *matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        T sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            sum += matrix[row * cols + i] * matrix[row * cols + i];
        }
        double rms = sqrt(sum / cols);
        for (int i = 0; i < cols; ++i) {
            matrix[row * cols + i] /= rms;
        }
    }
}

template <typename T>
void launchRmsNorm(T *matrix, int rows, int cols){
    dim3 blocksize(256);
    dim3 gridSize((rows + blocksize.x - 1) / blocksize.x);
    rmsNormalizationKernel<<<gridSize, blocksize>>>(matrix, rows, cols);
}


static PyObject* rms_norm(PyObject* self, PyObject* args) {
    PyArrayObject *input_matrix;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_matrix)) {
        return NULL;
    }

    double *matrix = static_cast<double*>(PyArray_DATA(input_matrix));
    int rows = PyArray_DIM(input_matrix, 0);
    int cols = PyArray_DIM(input_matrix, 1);

    if (dtype == NPY_FLOAT){
        float *matrix = static_cast<float*>(PyArray_DATA(input_matrix));
        float *d_matrix;
        cudaCheckError(cudaMalloc(&d_matrix, rows * cols * sizeof(float)));
        cudaCheckError(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        launchRmsNorm<float>(d_matrix, rows, cols);
        cudaCheckError(cudaMemcpy(matrix, d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaFree(d_matrix));
    }
    else if (dtype == NPY_DOUBLE){
        double *matrix = static_cast<double*>(PyArray_DATA(input_matrix));
        double *d_matrix;
        cudaCheckError(cudaMalloc(&d_matrix, rows * cols * sizeof(double)));
        cudaCheckError(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(double), cudaMemcpyHostToDevice));
        launchRmsNorm<double>(d_matrix, rows, cols);
        cudaCheckError(cudaMemcpy(matrix, d_matrix, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaFree(d_matrix));
    }
    else{
        PyErr_SetString(PyExc_TypeError, "Invalid data type");
        return NULL;
    }
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
