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
    // test with shared memory
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int idx = row * cols + tid;

    // use each thread to compute the square of each element
    float val = 0.0f;
    for(int i = tid; i < cols; i += blockDim.x) {
        float element = matrix[idx+i];
        sum += element*element;
    }
    sdata[tid] = sum;
    __syncthreads();

    // sum up the squares
    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid==0){
        float total_sum = sdata[0];
        flaot rms = sqrtf(total_sum/cols);
        sdata[0] = rms > 0.0f? rms: 1.0f;
    }
    __syncthreads();

    // normalize
    float rms = sdata[0];
    for (int i = tid; i < cols; i += blockDim.x) {
        matrix[idx+i] /= rms;
    }
}


static PyObject* rms_norm(PyObject* self, PyObject* args) {
    PyArrayObject *input_matrix;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_matrix)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be a NumPy array.");
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
    // int minGridSize, blockSize;
    // cudaCheckError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rmsNormalizationKernel, 0, rows * cols));
    // int gridSize = (rows + blockSize -1)/blockSize;
    // std::cout << "Optimal block size: " << blockSize << ", Grid size: " << gridSize << std::endl;
    int threadsPerBlock = (cols < 256)? cols: 256;
    int blocksPerGris = rows;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);


    //rmsNormalizationKernel<<<gridSize, blockSize>>>(d_matrix, rows, cols);
    rmsNormalizationKernel<<<blocksPerGris, threadsPerBlock, sharedMemSize>>>(d_matrix, rows, cols);
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
