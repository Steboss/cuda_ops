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
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int idx = row * cols + tid;

    // use each thread to compute the square of each element
    float val = 0.0f;
    if (tid < cols) {
        val = matrix[idx] * matrix[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    // sum up the squares
    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
        if(tid < s && (tid+s) < cols){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid==0){
        float sum = sdata[0];
        float rms = sqrt(sum / cols);
        sdata[0] = rms > 0.0f? rms: 1.0f;
    }
    __syncthreads();

    // normalize
    if (tid < cols){
        matrix[idx] /= sdata[0];
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
