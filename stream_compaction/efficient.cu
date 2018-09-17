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

// #define DEBUG 
		int paddedData[1 << 16];

		__global__ void kernUpSweep(int n, int* buffer, int interval) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index < n && index % interval == 0) {
				buffer[index + interval - 1] += buffer[index + (interval >> 1) - 1];
			}
		}

		__global__ void kernDownSweep(int n, int* buffer, int interval, int smallInterval) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index < n && index % interval == 0) {
				int t = buffer[index + smallInterval - 1];
				buffer[index + smallInterval - 1] = buffer[index + interval - 1];
				buffer[index + interval - 1] += t;
			}
		}

		__global__ void kernEfficientUpSweep(int n, int* buffer, int interval) {
			int index = (blockIdx.x * blockDim.x + threadIdx.x) * interval;
			if (index < n) {
				buffer[index + interval - 1] += buffer[index + (interval >> 1) - 1];
			}
		}

		__global__ void kernEfficientDownSweep(int n, int* buffer, int interval, int smallInterval) {
			int index = (blockIdx.x * blockDim.x + threadIdx.x) * interval;
			if (index < n) {
				int t = buffer[index + smallInterval - 1];
				buffer[index + smallInterval - 1] = buffer[index + interval - 1];
				buffer[index + interval - 1] += t;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool efficient) {
			// allocate 2 arrays on global memory. one original. one resized.
			int* dev_padded;

			int logn = ilog2ceil(n);
			int paddedSize = 1 << logn;
			size_t originalSizeInBytes = n * sizeof(int);
			size_t paddedSizeInBytes = paddedSize * sizeof(int);

			cudaMalloc((void**)&dev_padded, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_padded failed!");

			// initialize dev_padded to 0 and copy idata into it
			cudaMemset(dev_padded, 0, paddedSizeInBytes);
			checkCUDAError("cudaMemset failed");
			cudaMemcpy(dev_padded, idata, originalSizeInBytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcopy from idata to dev_original failed!");

			timer().startGpuTimer();

#ifdef DEBUG
				printf("before start: [");
				cudaMemcpy(paddedData, dev_padded, paddedSizeInBytes, cudaMemcpyDeviceToHost);
				for (int i = 0; i < paddedSize; i++) {
					printf("%d, ", paddedData[i]);
				}
				printf("] \n");
#endif

			scanHelper(paddedSize, logn, dev_padded, efficient);

			timer().endGpuTimer();


			// copy padded data back into odata
			cudaMemcpy(odata, dev_padded, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_original to odata failed!");

			// free the allocated arrays
			cudaFree(dev_padded);
			checkCUDAError("cudaFree on dev_padded failed");
        }


		/**
		* Helper function for scan that does the upsweeps and downsweeps
		*/
		void scanHelper(int n, int logn, int* dev_buffer, bool efficient) {
			
			if (efficient) {
				for (int i = 0; i <= logn - 1; i++) {
					int interval = 1 << (i + 1);
					dim3 numBlocks(((n >> (i + 1)) + Common::blockSize + 1) / Common::blockSize);
					kernEfficientUpSweep << < numBlocks, Common::blockSize >> > (n, dev_buffer, interval);
					checkCUDAError("kernEfficientUpSweep failed!");
				}
			}
			else {
				dim3 fullBlocksPerGrid((n + Common::blockSize - 1) / Common::blockSize);

				for (int i = 0; i <= logn - 1; i++) {
					int interval = 1 << (i + 1);
					kernUpSweep << < fullBlocksPerGrid, Common::blockSize >> > (n, dev_buffer, interval);
					checkCUDAError("kernEfficientUpSweep failed!");
				}
			}

#ifdef DEBUG
			printf("after up sweep: [");
			cudaMemcpy(paddedData, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i < n; i++) {
				printf("%d, ", paddedData[i]);
			}
			printf("] \n");
#endif

			// first set the last value to 0
			cudaMemset(dev_buffer + n - 1, 0, sizeof(int));
			checkCUDAError("kernSetIndexInData failed!");

#ifdef DEBUG
			printf("after setting last value: [");
			cudaMemcpy(paddedData, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i < n; i++) {
				printf("%d, ", paddedData[i]);
			}
			printf("] \n");
#endif

			// down sweep
			if (efficient) {
				for (int i = logn - 1; i >= 0; i--) {
					int smallInterval = 1 << i;
					int interval = 1 << (i + 1);
					dim3 numBlocks(((n >> (i + 1)) + Common::blockSize + 1) / Common::blockSize);
					kernEfficientDownSweep << < numBlocks, Common::blockSize >> > (n, dev_buffer, interval, smallInterval);
					checkCUDAError("kernEfficientDownSweep failed!");
				}
			}
			else {
				dim3 fullBlocksPerGrid((n + Common::blockSize - 1) / Common::blockSize);

				for (int i = logn - 1; i >= 0; i--) {
					int smallInterval = 1 << i;
					int interval = 1 << (i + 1);
					kernDownSweep << < fullBlocksPerGrid, Common::blockSize >> > (n, dev_buffer, interval, smallInterval);
					checkCUDAError("kernEfficientDownSweep failed!");
				}
			}

#ifdef DEBUG
			printf("after downsweep: [");
			cudaMemcpy(paddedData, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i < n; i++) {
				printf("%d, ", paddedData[i]);
			}
			printf("] \n");
#endif
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
        int compact(int n, int *odata, const int *idata, bool efficient) {
			if (n <= 0) return -1;

			int count = 0;

			// ----------------------------------------------
			// ---------- allocate global memory ------------
			// ----------------------------------------------

			int *dev_input, *dev_output, *dev_bools, *dev_indices;
			int logn = ilog2ceil(n);
			int paddedSize = 1 << logn;
			size_t originalSizeInBytes = n * sizeof(int);
			size_t paddedSizeInBytes = paddedSize * sizeof(int);
			dim3 fullBlocksPerGrid((paddedSize + Common::blockSize - 1) / Common::blockSize);

			cudaMalloc((void**)&dev_input, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_input failed!");
			cudaMalloc((void**)&dev_output, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_output failed!");
			cudaMalloc((void**)&dev_bools, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_indices failed!");

			// ----------------------------------------------------
			// ---------- copy data into global memory ------------
			// ----------------------------------------------------

			// set dev_input to 0 then copy idata into it
			cudaMemset(dev_input, 0, paddedSizeInBytes);
			checkCUDAError("cudaMemset failed");
			cudaMemcpy(dev_input, idata, originalSizeInBytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcopy from idata to dev_original failed!");

			// ---------------------------------------
			// ---------- start algorithm ------------
			// ---------------------------------------

			timer().startGpuTimer();
	
			// turn input into boolean array
			Common::kernMapToBoolean << <fullBlocksPerGrid, Common::blockSize >> > (paddedSize, dev_bools, dev_input);
			checkCUDAError("kernMapToBoolean failed!");

			// exclusive scan the boolean array
			cudaMemcpy(dev_indices, dev_bools, paddedSizeInBytes, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcopy from dev_bools to dev_indices failed!");
			scanHelper(paddedSize, logn, dev_indices, efficient);

			// scatter 
			Common::kernScatter << <fullBlocksPerGrid, Common::blockSize >> > (paddedSize, dev_output, dev_input, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();


			// -----------------------------------------------------------
			// ---------- read data from global memory ------------
			// -----------------------------------------------------------

			// first, copy dev_bool into odata to get the count of non-zero elements
			cudaMemcpy(odata, dev_bools, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_original to odata failed!");
			for (int i = 0; i < n; i++) {
				if (odata[i] != 0) count++;
			}

			// finally, copy dev_output into odata
			cudaMemcpy(odata, dev_output, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_original to odata failed!");

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
