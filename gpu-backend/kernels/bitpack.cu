// CUDA kernel: bit packing
//
// Compile: nvcc -ptx -o bitpack.ptx bitpack.cu

#include <stdint.h>

// Packs n indices into (n * bits_per_index / 8) bytes.
// Each thread handles ~8 indices for coalesced memory access.

extern "C" __global__ void bitpack_batch(
    const uint8_t* __restrict__ indices,   // raw indices (1 byte each)
    uint8_t* __restrict__ packed,           // packed output
    int num_indices,
    int bits_per_index
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread packs 8 consecutive indices
    for (int base = tid * 8; base < num_indices; base += stride * 8) {
        uint64_t accumulator = 0;
        int count = 0;
        for (int i = 0; i < 8 && base + i < num_indices; i++) {
            uint8_t idx = indices[base + i];
            accumulator |= ((uint64_t)(idx & ((1u << bits_per_index) - 1))) << (i * bits_per_index);
            count++;
        }

        int bit_offset = base * bits_per_index;
        int byte_offset = bit_offset / 8;
        int bit_shift = bit_offset % 8;

        // Write packed bytes (may span 9 bytes for 8 indices × 8 bits)
        for (int b = 0; b < (count * bits_per_index + 7) / 8; b++) {
            uint8_t byte_val = (uint8_t)((accumulator >> (b * 8 + bit_shift)) & 0xFF);
            // Atomic OR to handle overlapping writes from different threads
            atomicOr((int*)(&packed[byte_offset + b]), (int)byte_val);
        }
    }
}
