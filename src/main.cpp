/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 20; // feel free to change the size of array
const int NPOT = SIZE - 5; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];


void runtimeTest(int start, int end, int interval) {
	for (int i = start; i <= end; i+= interval) {
		int size = 1 << i;
		int npot = size - 7;

		int *array1 = new int[size];
		int *array2 = new int[size];
		int *array3 = new int[size];

		genArray(size - 1, array1, 50);  // Leave a 0 at the end to test that edge case
		array1[size - 1] = 0;

		printf("---------- array size is 2^%d ----------\n", i);
		printf("---------- scan ----------\n");

		zeroArray(size, array2);
		//printf("cpu scan, power-of-two: ");
		StreamCompaction::CPU::scan(size, array2, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("cpu scan, non power-of-two: ");
		StreamCompaction::CPU::scan(npot, array2, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("naive scan, power-of-two");
		StreamCompaction::Naive::scan(size, array3, array1);
		printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("naive scan, non-power-of-two");
		StreamCompaction::Naive::scan(npot, array3, array1);
		printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient scan, power-of-two, optimized thread number");
		StreamCompaction::Efficient::scan(size, array3, array1);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient scan, non-power-of-two, optimized thread number");
		StreamCompaction::Efficient::scan(size, array3, array1);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient scan, power-of-two, unoptimized thread number");
		StreamCompaction::Efficient::scan(size, array3, array1, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient scan, non-power-of-two, unoptimized thread number");
		StreamCompaction::Efficient::scan(npot, array3, array1, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("thrust scan, power-of-two");
		StreamCompaction::Thrust::scan(size, array3, array1);
		printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("thrust scan, non-power-of-two");
		StreamCompaction::Thrust::scan(npot, array3, array1);
		printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "");

		printf("---------- compact ----------\n");

		genArray(size - 1, array1, 4);  // Leave a 0 at the end to test that edge case
		array1[size - 1] = 0;

		zeroArray(size, array2);
		//printf("cpu compact without scan, power-of-two");
		StreamCompaction::CPU::compactWithoutScan(size, array2, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("cpu compact without scan, non-power-of-two");
		StreamCompaction::CPU::compactWithoutScan(npot, array3, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("cpu compact with scan, power of two");
		StreamCompaction::CPU::compactWithScan(size, array3, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("cpu compact with scan, non-power of two");
		StreamCompaction::CPU::compactWithScan(npot, array3, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("cpu compact with scan, power of two, optimized");
		StreamCompaction::CPU::compactWithScanOnePass(size, array3, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("cpu compact with scan, non-power of two, optimized");
		StreamCompaction::CPU::compactWithScanOnePass(npot, array3, array1);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient compact, power-of-two, optimized thread number");
		StreamCompaction::Efficient::compact(size, array3, array1);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient compact, non-power-of-two, otimized thread number");
		StreamCompaction::Efficient::compact(npot, array3, array1);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient compact, power-of-two, , unoptimized thread number");
		StreamCompaction::Efficient::compact(size, array3, array1, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");

		zeroArray(size, array3);
		//printf("work-efficient compact, non-power-of-two, unoptimized thread number");
		StreamCompaction::Efficient::compact(npot, array3, array1, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "");
		
		printf("\n\n\n");
		
		delete[] array1;
		delete[] array2;
		delete[] array3;
	}
	system("pause"); // stop Win32 console from closing on exit
}


int main(int argc, char* argv[]) {
	//runtimeTest(6, 22, 2);

    // Scan tests
	{
		printf("\n");
		printf("****************\n");
		printf("** SCAN TESTS **\n");
		printf("****************\n");

		genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		printArray(SIZE, a, true);

		// initialize b using StreamCompaction::CPU::scan you implement
		// We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
		// At first all cases passed because b && c are all zeroes.
		zeroArray(SIZE, b);
		printDesc("cpu scan, power-of-two");
		StreamCompaction::CPU::scan(SIZE, b, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//printArray(SIZE, b, true);

		zeroArray(SIZE, c);
		printDesc("cpu scan, non-power-of-two");
		StreamCompaction::CPU::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//printArray(NPOT, b, true);
		printCmpResult(NPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("naive scan, power-of-two");
		StreamCompaction::Naive::scan(SIZE, c, a);
		printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		// For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
		/*onesArray(SIZE, c);
		printDesc("1s array for finding bugs");
		StreamCompaction::Naive::scan(SIZE, c, a);
		printArray(SIZE, c, true); */

		zeroArray(SIZE, c);
		printDesc("naive scan, non-power-of-two");
		StreamCompaction::Naive::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(SIZE, c, true);
		printCmpResult(NPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient scan, power-of-two, optimized thread number");
		StreamCompaction::Efficient::scan(SIZE, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient scan, non-power-of-two, optimized thread number");
		StreamCompaction::Efficient::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient scan, power-of-two, unoptimized thread number");
		StreamCompaction::Efficient::scan(SIZE, c, a, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient scan, non-power-of-two, unoptimized thread number");
		StreamCompaction::Efficient::scan(NPOT, c, a, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("thrust scan, power-of-two");
		StreamCompaction::Thrust::scan(SIZE, c, a);
		printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		zeroArray(SIZE, c);
		printDesc("thrust scan, non-power-of-two");
		StreamCompaction::Thrust::scan(NPOT, c, a);
		printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);

		printf("\n");
		printf("*****************************\n");
		printf("** STREAM COMPACTION TESTS **\n");
		printf("*****************************\n");

		// Compaction tests

		genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		printArray(SIZE, a, true);

		int count, expectedCount, expectedNPOT;

		// initialize b using StreamCompaction::CPU::compactWithoutScan you implement
		// We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
		zeroArray(SIZE, b);
		printDesc("cpu compact without scan, power-of-two");
		count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		expectedCount = count;
		//printArray(count, b, true);
		printCmpLenResult(count, expectedCount, b, b);

		zeroArray(SIZE, c);
		printDesc("cpu compact without scan, non-power-of-two");
		count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		expectedNPOT = count;
		//printArray(count, c, true);
		printCmpLenResult(count, expectedNPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("cpu compact with scan");
		count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);

		zeroArray(SIZE, c);
		printDesc("cpu compact with scan and scatter in single loop");
		count = StreamCompaction::CPU::compactWithScanOnePass(SIZE, c, a);
		printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
		//printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient compact, power-of-two, optimized thread number");
		count = StreamCompaction::Efficient::compact(SIZE, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient compact, non-power-of-two, otimized thread number");
		count = StreamCompaction::Efficient::compact(NPOT, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(count, c, true);
		printCmpLenResult(count, expectedNPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient compact, power-of-two, , unoptimized thread number");
		count = StreamCompaction::Efficient::compact(SIZE, c, a, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);

		zeroArray(SIZE, c);
		printDesc("work-efficient compact, non-power-of-two, unoptimized thread number");
		count = StreamCompaction::Efficient::compact(NPOT, c, a, false);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(count, c, true);
		printCmpLenResult(count, expectedNPOT, b, c);

		system("pause"); // stop Win32 console from closing on exit
		delete[] a;
		delete[] b;
		delete[] c;
	}
}
