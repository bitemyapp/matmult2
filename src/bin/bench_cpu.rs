//! Reproducible end-to-end CPU benchmark. Inputs are built before timing;
//! allocation, packing, computation, and destruction of the result are timed.
use matmult2::*;
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let algorithm = args.get(1).map(String::as_str).unwrap_or("naive");
    let number = |i: usize, default| args.get(i).map(|s| s.parse::<usize>().unwrap()).unwrap_or(default);
    let n = number(2, 1024);
    let samples = number(3, 20);
    let threads = number(4, 4);
    let tile = number(5, 16);
    assert!(samples > 0);
    let mut a = FlatMatrix::new(n, n);
    let mut b = FlatMatrix::new(n, n);
    random_fill(&mut a, 1);
    random_fill(&mut b, 2);
    let multiply = || match algorithm {
        "naive" => naive_local_mult(black_box(&a), black_box(&b)),
        "tiled" => tiled_mult(black_box(&a), black_box(&b), tile),
        "parallel" => parallel_tiled_mult(black_box(&a), black_box(&b), tile, threads),
        "fused" => fused_mul_add(black_box(&a), black_box(&b)),
        _ => panic!("unknown algorithm: {algorithm}"),
    };
    let result = multiply();
    let checksum: f64 = result.as_slice().iter().sum();
    if std::env::var_os("VERIFY").is_some() {
        let reference = naive_local_mult(&a, &b);
        let max_error = result.as_slice().iter().zip(reference.as_slice())
            .map(|(x,y)| (x-y).abs()).fold(0.0_f64, f64::max);
        assert!(max_error <= 1e-11 * (n.max(1) as f64), "max error: {max_error}");
        eprintln!("verified all {} elements, max absolute error {max_error:e}", n*n);
    }
    drop(result);
    drop(black_box(multiply()));
    println!("algorithm,n,threads,tile,sample,ms,checksum");
    let mut times = Vec::with_capacity(samples);
    for sample in 0..samples {
        let start = Instant::now();
        drop(black_box(multiply()));
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        println!("{algorithm},{n},{threads},{tile},{sample},{ms:.6},{checksum:.17e}");
    }
    times.sort_by(f64::total_cmp);
    let percentile = |p: f64| times[((samples as f64 * p).ceil() as usize).saturating_sub(1)];
    let p50 = percentile(0.5);
    eprintln!("{algorithm} n={n} threads={threads} tile={tile}: median={p50:.3} ms p95={:.3} ms p99={:.3} ms min={:.3} ms max={:.3} ms GFLOP/s={:.2}",
        percentile(0.95), percentile(0.99), times[0], times[samples-1], 2.0*(n as f64).powi(3)/(p50*1e6));
}
