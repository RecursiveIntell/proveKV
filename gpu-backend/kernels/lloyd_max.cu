// CUDA kernel: batch Lloyd-Max scalar quantization
//
// Compile: nvcc -ptx -o lloyd_max.ptx lloyd_max.cu

#include <stdint.h>

// Pre-computed Gaussian centroids — constant memory
__constant__ float centroids_4[4];
__constant__ float centroids_8[8];
__constant__ float centroids_16[16];
__constant__ float centroids_32[32];

// Quantize a batch of scalar blocks.
// Each block of k scalars is normalized by its L2 norm,
// then each scalar is quantized against the nearest centroid.
//
// gridDim.x = num_vectors * (dim / k)
// blockDim.x = k (each thread handles one scalar in the block)

extern "C" __global__ void lloyd_max_encode(
    const float* __restrict__ vectors,  // [num_vectors × dim]
    uint8_t* __restrict__ indices,       // [num_vectors × (dim/k) × k]
    float* __restrict__ norms,           // [num_vectors × (dim/k)]
    int num_vectors,
    int dim,
    int k,
    int n_levels
) {
    int block_idx = blockIdx.x;  // which block (global index)
    int tid = threadIdx.x;       // which scalar within block

    int blocks_per_vector = dim / k;
    int vec_idx = block_idx / blocks_per_vector;
    int local_block = block_idx % blocks_per_vector;

    if (vec_idx >= num_vectors || tid >= k) return;

    int base = vec_idx * dim + local_block * k;

    // Compute block norm (shared across threads in block via shared memory)
    extern __shared__ float shared_block[];
    shared_block[tid] = vectors[base + tid];
    __syncthreads();

    // Warp reduction for norm
    float val = shared_block[tid];
    float norm_sq = val * val;

    // Simple reduction (works for k <= 32 which is typical)
    for (int offset = 1; offset < k; offset <<= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, norm_sq, offset);
        if (tid + offset < k) norm_sq += other;
    }

    // Broadcast norm from thread 0
    float norm;
    if (tid == 0) {
        norm = sqrtf(norm_sq);
        norms[block_idx] = norm;
    }
    norm = __shfl_sync(0xFFFFFFFF, norm, 0);

    // Quantize each scalar
    float normalized;
    if (norm > 1e-10f) {
        normalized = val / norm;
    } else {
        normalized = 0.0f;
        indices[base + tid] = 0;
        return;
    }

    // Find nearest centroid
    const float* centroids;
    switch (n_levels) {
        case 4:  centroids = centroids_4;  break;
        case 8:  centroids = centroids_8;  break;
        case 16: centroids = centroids_16; break;
        case 32: centroids = centroids_32; break;
        default: centroids = centroids_32; break;
    }

    float best_dist = 1e10f;
    uint8_t best_idx = 0;
    for (int c = 0; c < n_levels; c++) {
        float dist = fabsf(normalized - centroids[c]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = (uint8_t)c;
        }
    }

    indices[base + tid] = best_idx;
}

// Decode: indices + norms → reconstructed vectors
extern "C" __global__ void lloyd_max_decode(
    const uint8_t* __restrict__ indices,  // [num_vectors × (dim/k) × k]
    const float* __restrict__ norms,      // [num_vectors × (dim/k)]
    float* __restrict__ output,           // [num_vectors × dim]
    int num_vectors,
    int dim,
    int k,
    int n_levels
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int blocks_per_vector = dim / k;
    int vec_idx = block_idx / blocks_per_vector;
    int local_block = block_idx % blocks_per_vector;

    if (vec_idx >= num_vectors || tid >= k) return;

    int base = vec_idx * dim + local_block * k;
    float norm = norms[block_idx];
    int idx = indices[base + tid];

    const float* centroids;
    switch (n_levels) {
        case 4:  centroids = centroids_4;  break;
        case 8:  centroids = centroids_8;  break;
        case 16: centroids = centroids_16; break;
        case 32: centroids = centroids_32; break;
        default: centroids = centroids_32; break;
    }

    float centroid = centroids[idx];
    output[base + tid] = centroid * norm;
}
