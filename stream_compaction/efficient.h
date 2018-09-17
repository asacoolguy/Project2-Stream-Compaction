#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool efficient = true);

		void scanHelper(int n, int logn, int* dev_buffer, bool efficient);

		int compact(int n, int *odata, const int *idata, bool efficient = true);
    }
}
