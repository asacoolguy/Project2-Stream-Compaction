#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		__global__ void kernNaiveScan(int n, int* input, int* output, int interval) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				if (index < interval) {
					output[index] = input[index];
				}
				else if (index >= interval) {
					output[index] = input[index - interval] + input[index];
				}
			}
		}

		__global__ void kernMakeExclusive(int n, int* input, int* output) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n){
				if (index > 0) {
					output[index] = input[index - 1];
				}
				else if (index == 0){
					output[index] = 0;
				}
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			if (n <= 0) return;

			// allocate 2 arrays on global memory
			int* dev_buffer1;
			int* dev_buffer2;
			size_t sizeInBytes = n * sizeof(int);

			cudaMalloc((void**)&dev_buffer1, sizeInBytes);
			checkCUDAError("cudaMalloc dev_buffer1 failed!");

			cudaMalloc((void**)&dev_buffer2, sizeInBytes);
			checkCUDAError("cudaMalloc dev_buffer2 failed!");
			
			// copy the data into global memory
			cudaMemcpy(dev_buffer1, idata, sizeInBytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcopy from idata to dev_buffer1 failed!");
			cudaMemcpy(dev_buffer2, dev_buffer1, sizeInBytes, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcopy from dev_buffer1 to dev_buffer2 failed!");
			
			dim3 fullBlocksPerGrid((n + Common::blockSize - 1) / Common::blockSize);

			timer().startGpuTimer();
            
			int ceiling = ilog2ceil(n);
			int** input = &dev_buffer1;
			int** output = &dev_buffer2;

			for (int i = 1; i <= ceiling; i++) {
				int interval = 1 << (i - 1);
				kernNaiveScan << < fullBlocksPerGrid, Common::blockSize >> > (n, *input, *output, interval);
				checkCUDAError("kernNaiveScan failed");
				std::swap(input, output);

			}

			// shift the output array to the right by 1
			kernMakeExclusive << < fullBlocksPerGrid, Common::blockSize >> > (n, *input, *output);
			checkCUDAError("kernMakeExclusive failed");

            timer().endGpuTimer();

			// copy data back 
			cudaMemcpy(odata, *output, sizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from output to odata failed!");

			// free the allocated arrays
			cudaFree(dev_buffer1);
			checkCUDAError("cudaFree on dev_buffer1 failed");
			cudaFree(dev_buffer2);
			checkCUDAError("cudaFree on dev_buffer2 failed");
        }
    }
}
