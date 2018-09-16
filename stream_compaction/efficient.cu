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

		__global__ void kernEfficientUpSweep(int n, int* buffer, int interval) {
			// TODO: this interval thing is overloading the limit of int and causing it to roll back
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index < n && index % interval == 0) {
				buffer[index + interval - 1] += buffer[index + (interval >> 1) - 1];
			}
		}

		__global__ void kernEfficientDownSweep(int n, int* buffer, int interval, int smallInterval) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index < n && index % interval == 0) {
				int t = buffer[index + smallInterval - 1];
				buffer[index + smallInterval - 1] = buffer[index + interval - 1];
				buffer[index + interval - 1] += t;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
		// TODO: still has that bug where logn > 13 will crash some programs. maybe something wrong with cudaMalloc and memCpy?
        void scan(int n, int *odata, const int *idata) {
			// allocate 2 arrays on global memory. one original. one resized.
			int* dev_padded;
			int* dev_original;

			int logn = ilog2ceil(n);
			int paddedSize = 1 << logn;
			size_t originalSizeInBytes = n * sizeof(int);
			size_t paddedSizeInBytes = paddedSize * sizeof(int);

			dim3 fullBlocksPerGrid((paddedSize + Common::blockSize - 1) / Common::blockSize);

			cudaMalloc((void**)&dev_original, originalSizeInBytes);
			checkCUDAError("cudaMalloc dev_original failed!");
			cudaMalloc((void**)&dev_padded, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_padded failed!");

			// copy input into dev_original, then copy dev_original into dev_padded
			cudaMemcpy(dev_original, idata, originalSizeInBytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcopy from idata to dev_original failed!");
			Common::kernCopyArray << <fullBlocksPerGrid, Common::blockSize >> > (n, paddedSize, dev_original, dev_padded);
			checkCUDAError("kernCopyArray from dev_original to dev_padded failed!");

			timer().startGpuTimer();

#ifdef DEBUG
				printf("before start: [");
				cudaMemcpy(paddedData, dev_padded, paddedSizeInBytes, cudaMemcpyDeviceToHost);
				for (int i = 0; i < paddedSize; i++) {
					printf("%d, ", paddedData[i]);
				}
				printf("] \n");
#endif

			scanHelper(paddedSize, logn, fullBlocksPerGrid, dev_padded);

			timer().endGpuTimer();


			// copy padded data back into original data
			Common::kernCopyArray << <fullBlocksPerGrid, Common::blockSize >> > (paddedSize, n, dev_padded, dev_original);
			checkCUDAError("kernCopyArray from dev_padded to dev_original failed!");

			cudaMemcpy(odata, dev_original, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_original to odata failed!");

			// free the allocated arrays
			cudaFree(dev_original);
			checkCUDAError("cudaFree on dev_original failed");
			cudaFree(dev_padded);
			checkCUDAError("cudaFree on dev_padded failed");
        }


		// helper function for scan
		void scanHelper(int n, int logn, dim3 fullBlocksPerGrid, int* dev_buffer) {

			for (int i = 0; i <= logn - 1; i++) {
				int interval = 1 << (i + 1);
				kernEfficientUpSweep << < fullBlocksPerGrid, Common::blockSize >> > (n, dev_buffer, interval);
				checkCUDAError("kernEfficientUpSweep failed!");
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
			Common::kernSetIndexInData << <1, 1 >> > (n, n - 1, 0, dev_buffer);
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
			for (int i = logn - 1; i >= 0; i--) {
				int smallInterval = 1 << i;
				int interval = 1 << (i + 1);
				kernEfficientDownSweep << < fullBlocksPerGrid, Common::blockSize >> > (n, dev_buffer, interval, smallInterval);
				checkCUDAError("kernEfficientDownSweep failed!");
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
        int compact(int n, int *odata, const int *idata) {
			if (n <= 0) return -1;

			int count = 0;

			// ----------------------------------------------
			// ---------- allocate global memory ------------
			// ----------------------------------------------

			int *dev_original, *dev_input, *dev_output, *dev_bools, *dev_indices;
			int logn = ilog2ceil(n);
			int paddedSize = 1 << logn;
			size_t originalSizeInBytes = n * sizeof(int);
			size_t paddedSizeInBytes = paddedSize * sizeof(int);

			cudaMalloc((void**)&dev_original, originalSizeInBytes);
			checkCUDAError("cudaMalloc dev_original failed!");
			cudaMalloc((void**)&dev_input, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_input failed!");
			cudaMalloc((void**)&dev_output, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_output failed!");
			cudaMalloc((void**)&dev_bools, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, paddedSizeInBytes);
			checkCUDAError("cudaMalloc dev_indices failed!");

			dim3 fullBlocksPerGrid((paddedSizeInBytes + Common::blockSize - 1) / Common::blockSize);

			// ----------------------------------------------------
			// ---------- copy data into global memory ------------
			// ----------------------------------------------------

			// first copy the input into dev_original
			cudaMemcpy(dev_original, idata, originalSizeInBytes, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcopy from idata to dev_original failed!");

			// then copy the non-padded dev_original data into the padded dev_input
			Common::kernCopyArray << <fullBlocksPerGrid, Common::blockSize >> > (n, paddedSize, dev_original, dev_input);
			checkCUDAError("kernCopyArray from dev_original to dev_original failed!");

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
			scanHelper(paddedSize, logn, fullBlocksPerGrid, dev_indices);

			// scatter 
			Common::kernScatter << <fullBlocksPerGrid, Common::blockSize >> > (paddedSize, dev_output, dev_input, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();


			// -----------------------------------------------------------
			// ---------- read data from global memory ------------
			// -----------------------------------------------------------

			// first, copy dev_bool into dev_original and read it to get the count of non-zero elements. 
			Common::kernCopyArray << <fullBlocksPerGrid, Common::blockSize >> > (paddedSize, n, dev_bools, dev_original);
			checkCUDAError("kernCopyArray from dev_bools to dev_original failed!");
			cudaMemcpy(odata, dev_original, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_original to odata failed!");
			for (int i = 0; i < n; i++) {
				if (odata[i] != 0) count++;
			}

			// finally, copy dev_output into dev_original and read it to get the output
			Common::kernCopyArray << <fullBlocksPerGrid, Common::blockSize >> > (paddedSize, n, dev_output, dev_original);
			checkCUDAError("kernCopyArray from dev_output to dev_original failed!");
			cudaMemcpy(odata, dev_original, originalSizeInBytes, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcopy from dev_original to odata failed!");

			// ------------------------------------------
			// ---------- free global memory ------------
			// ------------------------------------------

			cudaFree(dev_original);
			checkCUDAError("cudaFree on dev_original failed");
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
