/*******************************************************************
*   CUDAKfNN.cu
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

#include "CUDAKfNN.h"

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDAKfNN_kernel(const cudaTextureObject_t tex_q, const int num_q, const cudaTextureObject_t tex_t, const int num_t, int* const __restrict__ g_match, const float threshold) {
	int ofs_t = threadIdx.x;
	float4 train = tex1Dfetch<float4>(tex_t, ofs_t);
	ofs_t += 32;
	float4 q[4];
	for (int i = 0, ofs_q = threadIdx.x + (blockIdx.x << 10) + (threadIdx.y << 7); i < 4; ++i, ofs_q += 32) q[i] = tex1Dfetch<float4>(tex_q, ofs_q);
	int best_i;
	float best_v = 10000000.0f, second_v = 20000000.0f;
#pragma unroll 6
	for (int t = 0; t < num_t; ++t, ofs_t += 32) {
		float dist[4];
		for (int i = 0; i < 4; ++i) {
			float tmp = q[i].w - train.w;
			dist[i] = tmp * tmp;
			tmp = q[i].x - train.x;
			dist[i] = fmaf(tmp, tmp, dist[i]);
			tmp = q[i].y - train.y;
			dist[i] = fmaf(tmp, tmp, dist[i]);
			tmp = q[i].z - train.z;
			dist[i] = fmaf(tmp, tmp, dist[i]);
		}
		for (int i = 0; i < 4; ++i) dist[i] += __shfl_xor(dist[i], 1);
		train = tex1Dfetch<float4>(tex_t, ofs_t);
		if (threadIdx.x & 1) dist[0] = dist[1];
		dist[0] += __shfl_xor(dist[0], 2);
		if (threadIdx.x & 1) dist[2] = dist[3];
		dist[2] += __shfl_xor(dist[2], 2);
		if (threadIdx.x & 2) dist[0] = dist[2];
		dist[0] += __shfl_xor(dist[0], 4);
		dist[0] += __shfl_xor(dist[0], 8);
		second_v = min(dist[0] += __shfl_xor(dist[0], 16), second_v);
		if (dist[0] < best_v) {
			second_v = best_v;
			best_i = t;
			best_v = dist[0];
		}
	}
	const int idx = (blockIdx.x << 5) + (threadIdx.y << 2) + threadIdx.x;
	if (idx < num_q && threadIdx.x < 4) g_match[idx] = best_v > threshold * second_v ? -1 : best_i;
}

void CUDAKfNN(const cudaTextureObject_t tex_t, const int num_t, const cudaTextureObject_t tex_q, const int num_q, int* const __restrict d_m, const float threshold) {
	CUDAKfNN_kernel<<<((num_q - 1) >> 5) + 1, { 32, 8 }>>>(tex_q, num_q, tex_t, num_t, d_m, threshold*threshold);
	cudaDeviceSynchronize();
}