/*******************************************************************
*   CUDAKfNN.h
*   CUDAKfNN
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 21, 2016
*******************************************************************/
//
// Fastest GPU implementation of a brute-force
// matcher for 128-float descriptors such as SIFT
// in 2NN mode, i.e., a match is returned if the best
// match between a query vector and a training vector
// is more than a certain threshold ratio
// better than the second-best match.
//
// Float descriptors are slow. Check out my CUDAK2NN project
// for much faster binary description matching. Use a
// good binary descriptor such as LATCH where possible.
//
// That said, this laboriously crafted kernel is EXTREMELY fast
// for a float matcher.
//
// CUDA CC 3.0 or higher is required.
//
// All functionality is contained in the files CUDAKfNN.h
// and CUDAKfNN.cu. 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#pragma once

#include <cstdint>

#include "cuda_runtime.h"

#ifdef __INTELLISENSE__
#define asm(x)
#define min(x) 0
#define fmaf(x) 0
#include "device_launch_parameters.h"
#define __CUDACC__
#include "device_functions.h"
#undef __CUDACC__
#endif

void CUDAKfNN(const cudaTextureObject_t tex_t, const int num_t, const cudaTextureObject_t tex_q, const int num_q, int* const __restrict d_m, const float threshold);
