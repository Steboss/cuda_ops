# Python and NumPy paths
PYTHON_INCLUDE := $(shell python3-config --includes)
NUMPY_INCLUDE := $(shell python3 -c "import numpy; print(numpy.get_include())")

# Compiler settings
NVCC := nvcc
NVCC_FLAGS := $(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -Xcompiler -fPIC -shared

# Compile
rms_norm.so: rms_norm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f *.so

.PHONY: clean
