// CUDA kernel: batch nearest-codeword index lookup
//
// For each (vector, sub-block) pair, find the codeword index c in [0, N) that
// minimizes the squared L2 distance between the sub-block and the codeword.
//
// Compile: nvcc -ptx -o codebook_lookup.ptx codebook_lookup.cu
//
// Layout assumptions (validated by Rust wrapper):
//   - input:     [n × d]           row-major f32, n vectors of dim d
//   - codebook:  [N × k]           row-major f32, N codewords of dim k
//   - output:    [n × block_count] row-major u32 indices
//   - k must be a small constant (4) for the inner loop to compile to a
//     tight unrolled sequence
//   - N must be <= 32 to fit in a single warp
//
// gridDim.x  = n * block_count
// blockDim.x = N (one thread per codeword; 32 is the sweet spot for warp)
//
// Per (vec, sub-block):
//   1. cooperatively load the k input floats into shared memory
//   2. each thread (codeword c) computes its squared L2 distance
//   3. warp butterfly finds the min
//   4. thread 0 writes the argmin to output
//
// Failure mode: if a codeword value is non-finite (NaN), its distance
// becomes NaN and the warp reduction can pick it. The Rust caller is
// expected to validate codebook finiteness at construction (which
// FibCodebookV1::build already does), so this is not a hot concern.

#include <stdint.h>

extern "C" __global__ void codebook_lookup_kernel(
    const float* __restrict__ input,    // [n × d]
    const float* __restrict__ codebook, // [N × k]
    uint32_t* __restrict__ output,      // [n × block_count]
    int n,
    int d,
    int k,
    int N
) {
    int block_idx = blockIdx.x;
    int c = threadIdx.x;  // codeword index

    if (block_idx >= n * (d / k)) return;

    int blocks_per_vector = d / k;
    int vec_idx = block_idx / blocks_per_vector;
    int sub_idx = block_idx % blocks_per_vector;

    // Load the k input floats into shared memory via threads 0..k-1.
    // For k=4 this is one warp's worth; pad with zeros for safety.
    extern __shared__ float shared_input[];
    if (c < k) {
        shared_input[c] = input[vec_idx * d + sub_idx * k + c];
    }
    __syncthreads();

    // Compute squared L2 distance from input sub-block to codeword c.
    // Unroll for k=4: 4 subtracts, 4 muls, 3 adds.
    float dist = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j < k) {
            float delta = shared_input[j] - codebook[c * k + j];
            dist += delta * delta;
        }
    }

    // Warp-wide argmin via butterfly. Each thread holds (c, dist) for its
    // own codeword; we want the c with the smallest dist.
    // For N <= 32 we can do it in log2(N) = 5 shuffle steps.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist = __shfl_xor_sync(0xFFFFFFFF, dist, offset);
        int   other_c    = __shfl_xor_sync(0xFFFFFFFF, c,    offset);
        if (other_dist < dist) {
            dist = other_dist;
            c    = other_c;
        }
    }

    if (threadIdx.x == 0) {
        output[block_idx] = (uint32_t)c;
    }
}
