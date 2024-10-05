#include <cmath>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


__global__ void rmsNormalizationKernel(float *matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            sum += matrix[row * cols + i] * matrix[row * cols + i];
        }
        float rms = sqrt(sum / cols);
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

    float *matrix = static_cast<float*>(PyArray_DATA(input_matrix));
    int rows = PyArray_DIM(input_matrix, 0);
    int cols = PyArray_DIM(input_matrix, 1);

    // Allocate GPU memory and copy data
    float *d_matrix;
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(256);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x);
    rmsNormalizationKernel<<<gridSize, blockSize>>>(d_matrix, rows, cols);

    // Copy result back to host
    cudaMemcpy(matrix, d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

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

PyMODINIT_FUNC PyInit_cuda_ops_rms_norm(void) {
    import_array();
    return PyModule_Create(&module);
}
