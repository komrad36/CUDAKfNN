/*******************************************************************
*   main.cpp
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace std::chrono;

struct Match {
	int q, t;
	Match() {}
	Match(const int _q, const int _t) : q(_q), t(_t) {}
};

int main() {
	// ------------- Configuration ------------
	constexpr int warmups = 10;
	constexpr int runs = 10;
	constexpr int size = 10000;
	constexpr float threshold = 0.98f;
	// --------------------------------


	// ------------- Generation of Random Data ------------
	// obviously, this is not representative of real data;
	// it doesn't matter for brute-force matching
	// but the MIH methods will be much faster
	// on real data
	std::cout << std::endl << "Generating random test data..." << std::endl;
	std::mt19937 gen(std::mt19937::default_seed);
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);
	float* fqvecs = reinterpret_cast<float*>(malloc(128 * sizeof(float) * size));
	float* ftvecs = reinterpret_cast<float*>(malloc(128 * sizeof(float) * size));
	uint8_t* uqvecs = reinterpret_cast<uint8_t*>(malloc(128 * sizeof(uint8_t) * size));
	uint8_t* utvecs = reinterpret_cast<uint8_t*>(malloc(128 * sizeof(uint8_t) * size));
	for (int i = 0; i < 128 * size; ++i) {
		fqvecs[i] = dis(gen);
		uqvecs[i] = static_cast<uint8_t>(255.0f*fqvecs[i] + 0.5f);
		ftvecs[i] = dis(gen);
		utvecs[i] = static_cast<uint8_t>(255.0f*ftvecs[i] + 0.5f);
	}
	// --------------------------------

	// setting cache and shared modes
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// allocating and transferring query vecs and binding to texture object
	void* d_qvecs;
	cudaMalloc(&d_qvecs, 128 * sizeof(float) * size);
	cudaMemcpy(d_qvecs, fqvecs, 128 * sizeof(float) * size, cudaMemcpyHostToDevice);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_qvecs;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = resDesc.res.linear.desc.y = resDesc.res.linear.desc.z  = resDesc.res.linear.desc.w = 32;
	resDesc.res.linear.sizeInBytes = 128 * sizeof(float) * size;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] = texDesc.addressMode[3] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t tex_q = 0;
	cudaCreateTextureObject(&tex_q, &resDesc, &texDesc, nullptr);

	// allocating and transferring training vecs and binding to texture object
	// NOTE: always allocate 8 EXTRA AT THE END. Contents don't
	// matter but the allocation does.
	void* d_tvecs;
	cudaMalloc(&d_tvecs, 128 * sizeof(float) * (size + 8));
	cudaMemcpy(d_tvecs, ftvecs, 128 * sizeof(float) * size, cudaMemcpyHostToDevice);
	struct cudaResourceDesc resDesc_train;
	memset(&resDesc_train, 0, sizeof(resDesc_train));
	resDesc_train.resType = cudaResourceTypeLinear;
	resDesc_train.res.linear.devPtr = d_tvecs;
	resDesc_train.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc_train.res.linear.desc.x = resDesc_train.res.linear.desc.y = resDesc_train.res.linear.desc.z = resDesc_train.res.linear.desc.w = 32;
	resDesc_train.res.linear.sizeInBytes = 128 * sizeof(float) * size;
	struct cudaTextureDesc texDesc_train;
	memset(&texDesc_train, 0, sizeof(texDesc_train));
	texDesc_train.addressMode[0] = texDesc_train.addressMode[1] = texDesc_train.addressMode[2] = texDesc_train.addressMode[3] = cudaAddressModeBorder;
	texDesc_train.filterMode = cudaFilterModePoint;
	texDesc_train.readMode = cudaReadModeElementType;
	texDesc_train.normalizedCoords = 0;
	cudaTextureObject_t tex_t = 0;
	cudaCreateTextureObject(&tex_t, &resDesc_train, &texDesc_train, nullptr);

	// allocating space for match results
	int* d_matches;
	cudaMalloc(&d_matches, 4 * size);

	std::cout << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) CUDAKfNN(tex_t, size, tex_q, size, d_matches, threshold);
	std::cout << "Testing..." << std::endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CUDAKfNN(tex_t, size, tex_q, size, d_matches, threshold);
	high_resolution_clock::time_point end = high_resolution_clock::now();
	// --------------------------------


	// transferring matches back to host
	int* h_matches = reinterpret_cast<int*>(malloc(sizeof(int) * size));
	cudaMemcpy(h_matches, d_matches, sizeof(int) * size, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::vector<Match> matches;
	for (int i = 0; i < size; ++i) {
		if (h_matches[i] != -1) matches.emplace_back(i, h_matches[i]);
	}

	double total = 0.0;
	for (auto& m : matches) {
		for (int i = 0; i < 128; ++i) {
			total += static_cast<double>(ftvecs[(m.t << 7) + i]) + static_cast<double>(fqvecs[(m.q << 7) + i]);
		}
	}
	std::cout.precision(17);
	std::cout << "Checksum: " << total << std::endl;
	std::cout.precision(-1);
	//if (total != 358851.69540632586) {
	//	for (int i = 0; i < 5; ++i) std::cout << "ERROR!" << std::endl;
	//}

	const double sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	std::cout << std::endl << "CUDAKfNN found " << matches.size() << " matches in " << sec * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(size)*static_cast<double>(size) / sec * 1e-9 << " billion comparisons/second." << std::endl << std::endl;
}
