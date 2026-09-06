//! Hardware-in-loop receipt for the activated CUDA codebook lookup path.
//!
//! Run only on an NVIDIA CUDA host:
//! `cargo run --release -p gpu-backend --example gpu_activation_receipt --features gpu --locked`
//!
//! The program exits nonzero rather than representing CPU fallback as GPU proof.

use std::process;
use std::time::Instant;

use gpu_backend::{
    codebook_kernel_source_digest, codebook_lookup_batch, fallback::codebook_lookup_cpu,
    GpuAvailability, GpuContext,
};

const SCHEMA: &str = "GpuActivationReceiptV1";
const N: usize = 32;
const D: usize = 64;
const K: usize = 4;
const CODEWORDS: usize = 32;

fn json_string(value: &str) -> String {
    format!("\"{}\"", value.escape_default())
}

fn unavailable_receipt(availability: &GpuAvailability) {
    let detail = availability
        .detail()
        .map(json_string)
        .unwrap_or_else(|| "null".to_owned());
    println!(
        "{{\"schema\":\"{SCHEMA}\",\"outcome\":\"blocked\",\"availability\":{},\"detail\":{detail},\"source_digest\":{}}}",
        json_string(availability.kind()),
        json_string(&codebook_kernel_source_digest()),
    );
}

fn main() {
    let availability = GpuContext::availability();
    if !matches!(availability, GpuAvailability::Ready) {
        unavailable_receipt(&availability);
        process::exit(2);
    }

    let context = GpuContext::init().expect("ready availability must provide CUDA context");
    let codebook: Vec<f32> = (0..CODEWORDS * K)
        .map(|index| (index as f32 - 64.0) / 19.0)
        .collect();
    let input: Vec<f32> = (0..N)
        .flat_map(|_| codebook[5 * K..6 * K].iter().copied().cycle().take(D))
        .collect();

    let cpu_started = Instant::now();
    let cpu = codebook_lookup_cpu(&input, &codebook, N, D, K)
        .expect("CPU reference must accept the receipt fixture");
    let cpu_elapsed = cpu_started.elapsed();

    let gpu_started = Instant::now();
    let gpu = codebook_lookup_batch(&input, &codebook, N, D, K)
        .expect("ready CUDA path must execute the receipt fixture");
    let gpu_elapsed = gpu_started.elapsed();

    let mismatch_count = cpu
        .iter()
        .zip(&gpu)
        .filter(|(left, right)| left != right)
        .count();
    let outcome = if mismatch_count == 0 {
        "verified"
    } else {
        "parity_failed"
    };
    println!(
        "{{\"schema\":\"{SCHEMA}\",\"outcome\":\"{outcome}\",\"availability\":\"ready\",\"device_name\":{},\"memory_bytes\":{},\"source_digest\":{},\"shape\":{{\"n\":{N},\"d\":{D},\"k\":{K},\"codewords\":{CODEWORDS}}},\"index_count\":{},\"mismatch_count\":{mismatch_count},\"cpu_elapsed_ns\":{},\"gpu_elapsed_ns\":{}}}",
        json_string(&context.device_name),
        context.memory_bytes,
        json_string(&codebook_kernel_source_digest()),
        gpu.len(),
        cpu_elapsed.as_nanos(),
        gpu_elapsed.as_nanos(),
    );

    if mismatch_count != 0 {
        process::exit(3);
    }
}
