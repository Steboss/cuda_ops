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
        T rms = sqrt(sum / cols);
        // avoid division by zero
        if (rms > 0.0){
            for (int i = 0; i < cols; ++i) {
                matrix[row * cols + i] /= rms;
            }
        } else {
            for (int i = 0; i < cols; ++i) {
                matrix[row * cols + i] = 0.0;
            }
        }
    }
}

template <typename T>
void launchRmsNorm(T *matrix, int rows, int cols){
    dim3 blocksize(256);
    dim3 gridSize((rows + blocksize.x - 1) / blocksize.x);
    rmsNormalizationKernel<<<gridSize, blocksize>>>(matrix, rows, cols);
}


template <typename T>
void handleRmsNorm(T *matrix, int rows, int cols){
    T *d_matrix;
    cudaStream_t stream;
    cudaCheckError(cudaStreamCreate(&stream));
    cudaCheckError(cudaMalloc((void**)&d_matrix, rows * cols * sizeof(T)));
    cudaCheckError(cudaMemcpyAsync(d_matrix, matrix, rows * cols * sizeof(T), cudaMemcpyHostToDevice, stream));
    launchRmsNorm<T>(d_matrix, rows, cols);
    cudaCheckError(cudaMemcpyAsync(matrix, d_matrix, rows * cols * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaStreamSynchronize(stream));
    cudaCheckError(cudaFree(d_matrix));
    cudaCheckError(cudaStreamDestroy(stream));
}

static PyObject* rms_norm(PyObject* self, PyObject* args) {
    PyArrayObject *input_matrix;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_matrix)) {
        return NULL;
    }

    double *matrix = static_cast<double*>(PyArray_DATA(input_matrix));
    int rows = PyArray_DIM(input_matrix, 0);
    int cols = PyArray_DIM(input_matrix, 1);
    int dtype = PyArray_TYPE(input_matrix);

    if (dtype == NPY_FLOAT){
        float *matrix = static_cast<float*>(PyArray_DATA(input_matrix));
        handleRmsNorm<float>(matrix, rows, cols);
    }
    else if (dtype == NPY_DOUBLE){
        double *matrix = static_cast<double*>(PyArray_DATA(input_matrix));
        handleRmsNorm<double>(matrix, rows, cols);
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
