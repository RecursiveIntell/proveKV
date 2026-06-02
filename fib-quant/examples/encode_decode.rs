use fib_quant::{FibQuantProfileV1, FibQuantizer};

fn main() -> fib_quant::Result<()> {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 42)?;
    profile.training_samples = 128;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 2;

    let quantizer = FibQuantizer::new(profile)?;
    let input = vec![1.0, 0.5, -0.25, 0.125, -1.5, 0.75, 0.25, -0.5];
    let (code, receipt) = quantizer.encode_with_receipt(&input)?;
    let decoded = quantizer.decode(&code)?;

    println!("encoded_digest={}", receipt.encoded_digest);
    println!("source_vector_digest={}", receipt.source_vector_digest);
    println!("decoded_len={}", decoded.len());
    Ok(())
}
