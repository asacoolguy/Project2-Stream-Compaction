#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

		void scanHelper(int n, int logn, dim3 fullBlocksPerGrid, int* dev_buffer);

        int compact(int n, int *odata, const int *idata);
    }
}
