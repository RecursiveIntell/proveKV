use fib_quant::{FibCodebookV1, FibQuantProfileV1};

fn main() -> fib_quant::Result<()> {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 42)?;
    profile.training_samples = 128;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 2;

    let codebook = FibCodebookV1::build(profile)?;
    println!("{}", codebook.codebook_digest);
    Ok(())
}
