#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            
			if (n > 0) {
				odata[0] = 0;

				for (int i = 1; i < n; i++) {
					odata[i] = idata[i - 1] + odata[i - 1];
				}
			}

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            
			int count = 0;

			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[count++] = idata[i];
				}
			}

	        timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        
			int count = 0;

			if (n > 0) {
				
				// first scan
				odata[0] = 0;
				for (int i = 1; i < n; i++) {
					odata[i] = (idata[i - 1] != 0 ? 1 : 0) + odata[i - 1];
				}

				// then scatter
				for (int i = 0; i < n; i++) {
					if (idata[i] != 0) {
						odata[odata[i]] = idata[i];
						count++;
					}
				}
			}

	        timer().endCpuTimer();
            return count;
        }

		/**
		* CPU stream compaction using scan and scatter in a single for loop
		*
		* @returns the number of elements remaining after compaction.
		*/
		int compactWithScanOnePass(int n, int *odata, const int *idata) {
			timer().startCpuTimer();

			int count = 0;

			if (n > 0) {
				odata[0] = 0;
				for (int i = 0; i < n; i++) {
					// first scan
					if (i < n - 1) {
						odata[i + 1] = (idata[i] != 0 ? 1 : 0) + odata[i];
					}
					// then scatter
					if (idata[i] != 0) {
						odata[odata[i]] = idata[i];
						count++;
					}
				}
			}

			timer().endCpuTimer();
			return count;
		}
    }
}
