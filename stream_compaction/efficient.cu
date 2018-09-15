#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernEfficientUpSweep(int n, int* buffer, int interval) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x * interval;
			if (index < n) {
				buffer[index + interval - 1] += buffer[index + (interval >> 1) - 1];
			}
		}

		__global__ void kernEfficientDownSweep(int n, int* buffer, int interval) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x * interval;

			if (index < n) {
				int smallInterval = interval >> 1;
				int t = buffer[index + smallInterval - 1];
				buffer[index + smallInterval - 1] = buffer[index + interval - 1];
				buffer[index + interval - 1] += t;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		// TODO: for somereason this fails for 2^n where n >= 9. weird. am i doing memcpy wrong?
        void scan(int n, int *odata, const int *idata) {
			// allocate buffer array on global memory
			int* dev_original;
			int* dev_buffer;

			int size = 1 << (ilog2ceil(n));
			size_t originalSizeInBytes = n * sizeof(int);
			size_t sizeInBytes = size * sizeof(int);
			dim3 fullBlocksPerGrid((size + Common::blockSize - 1) / Common::blockSize);

			cudaMalloc((void**)&dev_original, originalSizeInBytes);
			checkCUDAError("cudaMalloc dev_original failed!");
			cudaMalloc((void**)&dev_buffer, sizeInBytes);
			checkCUDAError("cudaMalloc dev_buffer failed!");

			// copy the data into global memory
			cudaMemcpy(dev_original, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcopy from idata to dev_original failed!");

			// copy the original data into the buffer
			Common::kernCopyArray << <fullBlocksPerGrid, Common::blockSize >> >(n, size, dev_original, dev_buffer);
			checkCUDAError("kernCopy from dev_original to dev_buffer failed!");

			timer().startGpuTimer();

			//printf("before start: [");
			//cudaMemcpy(odata, dev_buffer, sizeInBytes, cudaMemcpyDeviceToHost);
			//for (int i = 0; i < size; i++) {
			//	printf("%d, ", odata[i]);
			//}
			//printf("] \n");

			scanHelper(n, size, dev_buffer);

			timer().endGpuTimer();

			// copy the buffer data back into the original
			Common::kernCopyArray << <fullBlocksPerGrid, Common::blockSize >> >(size, n, dev_buffer, dev_original);
			checkCUDAError("kernCopy from dev_buffer to dev_original failed!");

			// copy data back 
			cudaMemcpy(odata, dev_original, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_original to odata failed!");

			// free the allocated arrays
			cudaFree(dev_original);
			checkCUDAError("cudaFree on dev_original failed");
			cudaFree(dev_buffer);
			checkCUDAError("cudaFree on dev_buffer failed");
        }


		// helper function for scan
		void scanHelper(int n, int size, int* dev_buffer) {
			dim3 fullBlocksPerGrid((size + Common::blockSize - 1) / Common::blockSize);

			for (int i = 0; i <= ilog2ceil(n) - 1; i++) {
				int interval = 1 << (i + 1);
				kernEfficientUpSweep << < fullBlocksPerGrid, Common::blockSize >> > (size, dev_buffer, interval);
				checkCUDAError("kernEfficientUpSweep failed!");
			}

			// first set the last value to 0
			Common::kernSetIndexInData << <1, 1 >> > (size, size - 1, 0, dev_buffer);
			checkCUDAError("kernSetIndexInData failed!");

			// down sweep
			for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
				int interval = 1 << (i + 1);
				kernEfficientDownSweep << < fullBlocksPerGrid, Common::blockSize >> > (size, dev_buffer, interval);
				checkCUDAError("kernEfficientDownSweep failed!");
			}
		}


        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
			if (n <= 0) return -1;

			int count = 0;

			// ----------------------------------------------
			// ---------- allocate global memory ------------
			// ----------------------------------------------

			int* dev_input;
			int* dev_output;
			int* dev_bools;
			int* dev_indices;
			int size = 1 << (ilog2ceil(n));
			size_t originalSizeInBytes = n * sizeof(int);
			size_t sizeInBytes = size * sizeof(int);

			cudaMalloc((void**)&dev_input, sizeInBytes);
			checkCUDAError("cudaMalloc dev_input failed!");
			cudaMalloc((void**)&dev_output, sizeInBytes);
			checkCUDAError("cudaMalloc dev_output failed!");
			cudaMalloc((void**)&dev_bools, sizeInBytes);
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, sizeInBytes);
			checkCUDAError("cudaMalloc dev_indices failed!");

			cudaDeviceSynchronize();

			dim3 fullBlocksPerGrid((size + Common::blockSize - 1) / Common::blockSize);

			// ----------------------------------------------------
			// ---------- copy data into global memory ------------
			// ----------------------------------------------------

			// copy the input into global memory
			cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcopy from idata to dev_original failed!");

			// ---------------------------------------
			// ---------- start algorithm ------------
			// ---------------------------------------

			timer().startGpuTimer();
	
			// turn input into boolean array
			Common::kernMapToBoolean << <fullBlocksPerGrid, Common::blockSize >> > (size, dev_bools, dev_input);
			checkCUDAError("kernMapToBoolean failed!");

			// exclusive scan the boolean array
			cudaMemcpy(dev_indices, dev_bools, sizeInBytes, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcopy from dev_bools to dev_indices failed!");
			scanHelper(n, size, dev_indices);

			// scatter 
			Common::kernScatter << <fullBlocksPerGrid, Common::blockSize >> > (n, dev_output, dev_input, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();
			// -----------------------------------------
			// ------------- end algorithm -------------
			// -----------------------------------------


			// -----------------------------------------------------------
			// ---------- read global memory into host memory ------------
			// -----------------------------------------------------------

			// first, read dev_bool to get the count of non-zero elements
			cudaMemcpy(odata, dev_bools, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_output to odata failed!");
			for (int i = 0; i < n; i++) {
				if (odata[i] != 0) count++;
			}

			// finally, get the values out of the original array
			cudaMemcpy(odata, dev_output, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_output to odata failed!");

			// ------------------------------------------
			// ---------- free global memory ------------
			// ------------------------------------------

			cudaFree(dev_input);
			checkCUDAError("cudaFree on dev_input failed");
			cudaFree(dev_output);
			checkCUDAError("cudaFree on dev_output failed");
			cudaFree(dev_bools);
			checkCUDAError("cudaFree on dev_bools failed");
			cudaFree(dev_indices);
			checkCUDAError("cudaFree on dev_indices failed");

			return count;
        }
    }
}
