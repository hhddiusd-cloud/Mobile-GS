/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}









__global__ void
renderNoSortKernel( int P,int W, int H,
    const float2* __restrict__ means2D,
    const glm::vec3* scales,
    const float* __restrict__ feature,
    const float* __restrict__ depths,
    const float* __restrict__ thetas,
    const float* __restrict__ phis,
    float* __restrict__ w_fg,
    const float4* __restrict__ conic_opacity,
    const int* __restrict__ radii,
    float* __restrict__ out_color,
    float* __restrict__ Ts)
    {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= P) return;


    int r = radii[idx];
    if (r <= 0) return;

    float2 center = means2D[idx];
    float4 co = conic_opacity[idx];


    if (co.w == 0.0f) return;



    float  depth  = depths[idx];
    if (depth < 0) return;
    depth = fmaxf(depth, 1e-6);

    float  theta  = thetas[idx];
    float  phi    = phis[idx];
    glm::vec3  scale    = scales[idx];

    float max_scale = fmaxf(scale.x, fmaxf(scale.y, scale.z));
    float weight = expf(-max_scale / depth ) + phi / (depth * depth) + phi*phi ;

 
    // NUM_CHANNELS=3（RGB）
    float c0 = feature[idx * NUM_CHANNELS + 0];
    float c1 = feature[idx * NUM_CHANNELS + 1];
    float c2 = feature[idx * NUM_CHANNELS + 2];

    // bounding box
    // [center.x-r, center.x+r], [center.y-r, center.y+r]
    int x0 = max(0,        int(floorf(center.x - r)) );
    int x1 = min(W - 1,    int( ceilf(center.x + r)) );
    int y0 = max(0,        int(floorf(center.y - r)) );
    int y1 = min(H - 1,    int( ceilf(center.y + r)) );


    for (int py = y0; py <= y1; ++py)
    {
        for (int px = x0; px <= x1; ++px)
        {
            float dx = (center.x - px);
            float dy = (center.y - py);
            // power = -0.5 * (xx dx^2 + yy dy^2) - xy dx dy
            float power = -0.5f * (co.x * dx * dx + co.z * dy * dy) - co.y * dx * dy;
            if (power > 0.f)
                continue;

            float alpha = fminf(0.99f, co.w * __expf(power));
            if (alpha < 1.f / 255.f) {
                continue;
            }
            int pixel_id = py * W + px;



            float logTerm = logf(fmaxf(1.0f - alpha, 1e-6f));   // avoid log(0)
            atomicAdd(&Ts[pixel_id], logTerm);


            atomicAdd(&out_color[0 * W * H + pixel_id], c0 * alpha * weight);
            atomicAdd(&out_color[1 * W * H + pixel_id], c1 * alpha * weight);
            atomicAdd(&out_color[2 * W * H + pixel_id], c2 * alpha * weight);

            atomicAdd(&w_fg[pixel_id], alpha * weight);

               }
    }

 }


__global__ void composeKernel( int W, int H,
    float* __restrict__ out_color,
    float* __restrict__ w_fg,
    const float* __restrict__ bg_color,
    const float* __restrict__ Ts
    )
    {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= W * H) return;


    float Wfg  = w_fg[ idx] + 1e-6f;
    float T  = __expf(Ts[idx]);

    for (int ch = 0; ch < NUM_CHANNELS; ch++)
    {
        float C    = out_color[ch * W * H + idx];
        float Cbg  = bg_color[ch]; 

        out_color[ch * W * H + idx] = C / Wfg * (1.0f-T) + T * Cbg;


    }
    }











// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* theta,
    const float* phi,
    float* w_fg,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* Ts,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
// 	size_t img_chunk_size = required<ImageState>(width * height);
// 	char* img_chunkptr = imageBuffer(img_chunk_size);
// 	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)




    int blockSize = 256;
    int gridSize  = (P + blockSize - 1) / blockSize;
    renderNoSortKernel<<<gridSize, blockSize>>>(
        P, width, height,
        geomState.means2D,
        (glm::vec3*)scales,
        geomState.rgb,
        geomState.depths,
        theta,
        phi,
        w_fg,
        geomState.conic_opacity,
        radii,
        out_color,
        Ts
    );

    int gridSize_img  = (width * height + blockSize - 1) / blockSize;

    composeKernel<<<gridSize_img, blockSize>>>(
        width, height,
        out_color,
        w_fg,
        background,
        Ts
    );
}

