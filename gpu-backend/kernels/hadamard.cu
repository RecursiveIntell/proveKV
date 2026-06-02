// CUDA kernel: batch Hadamard Walsh-Hadamard Transform
//
// Compile: nvcc -ptx -o hadamard.ptx hadamard.cu
// Load in Rust: let ptx = include_str!("../kernels/hadamard.ptx");
// let module = ctx.load_module(ptx)?;

// Each block handles one vector.
// gridDim.x = num_vectors
// blockDim.x = dim (must be <= 1024 and a power of 2)
extern "C" __global__ void hadamard_wht_batch(
    float* data,             // [num_vectors × dim], in-place
    const int* signs,        // [dim] random ±1 per vector
    int num_vectors,
    int dim
) {
    int vec_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (vec_idx >= num_vectors || tid >= dim) return;

    int offset = vec_idx * dim;

    // Load into shared memory for the WHT butterfly
    extern __shared__ float shared_data[];
    shared_data[tid] = data[offset + tid];
    __syncthreads();

    // In-place WHT butterfly
    for (int step = 1; step < dim; step <<= 1) {
        int pair_idx = tid;
        int partner = pair_idx ^ step;

        if (tid < (pair_idx & ~(step - 1)) + step && partner < dim) {
            float a = shared_data[tid];
            float b = shared_data[partner];
            shared_data[tid] = a + b;
            shared_data[partner] = a - b;
        }
        __syncthreads();
    }

    // Apply signs and scale
    float scale = rsqrtf((float)dim);
    data[offset + tid] = shared_data[tid] * signs[tid] * scale;
}
