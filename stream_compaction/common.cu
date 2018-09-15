#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

		/**
		* Sets a single value in data to a certain value
		*/
		__global__ void kernSetIndexInData(int n, int index, int value, int *data) {
			if (index >= 0 && index < n) {
				data[index] = value;
			}
		}

		/**
		* Copies elements from array1 into array2. 
		* If array 1 has less elements, the empty elements in array 2 will be 0.
		* If array 1 has more elements, the extra elements will not be copied over.
		*/
		__global__ void kernCopyArray(int size1, int size2, int * array1, int * array2)	{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < size2) {
				if (index < size1) {
					array2[index] = array1[index];
				}
				else {
					array2[index] = 0;
				}
			}
		}

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				bools[index] = idata[index] != 0 ? 1 : 0;
			}
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				if (bools[index] == 1) {
					odata[indices[index]] = idata[index];
				}
			}
        }

    }
}
