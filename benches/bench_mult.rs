use divan::{Bencher, black_box};
use matmult2::FlatMatrix;
use matmult2::fused_mul_add;
use matmult2::naive_local_mult;
use matmult2::naive_mult;
use matmult2::parallel_tiled_mult;
use matmult2::random_fill;
use matmult2::tiled_mult;
use matmult2::{CpuMultiplier, fast_mult};

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench]
fn fast_cpu(bencher: Bencher) {
    let (a, b) = setup(MAT_SIZE);
    black_box(fast_mult(&a, &b)); // Initialize persistent CPU workers before timing.
    bencher.bench_local(|| {
        black_box(fast_mult(&a, &b));
    });
}

#[divan::bench(args = [1, 4, 12, 24])]
fn packed_gemm(bencher: Bencher, threads: usize) {
    let (a, b) = setup(MAT_SIZE);
    let engine = CpuMultiplier::new(threads);
    black_box(engine.multiply(&a, &b));
    bencher.bench_local(|| {
        black_box(engine.multiply(&a, &b));
    });
}

#[divan::bench(args = [1, 4, 12, 32])]
fn packed_ordered(bencher: Bencher, threads: usize) {
    let (a, b) = setup(MAT_SIZE);
    let engine = CpuMultiplier::new(threads);
    black_box(engine.multiply_ordered(&a, &b));
    bencher.bench_local(|| {
        black_box(engine.multiply_ordered(&a, &b));
    });
}

fn setup(size: usize) -> (FlatMatrix, FlatMatrix) {
    let mut a = FlatMatrix::new(size, size);
    let mut b = FlatMatrix::new(size, size);
    random_fill(&mut a, 1);
    random_fill(&mut b, 2);
    (a, b)
}

const MAT_SIZE: usize = 1024;
const TILE_SIZES: &[usize] = &[8, 16, 32, 64, 128, 256, 512];
const THREAD_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8];

fn cart_prod(v1: &[usize], v2: &[usize]) -> Vec<(usize, usize)> {
    v1.iter()
        .flat_map(|&x| v2.iter().map(move |&y| (x, y)))
        .collect()
}

#[divan::bench]
#[ignore]
fn naive(bencher: Bencher) {
    let (a, b) = setup(MAT_SIZE);
    bencher.bench_local(|| {
        black_box(naive_mult(&a, &b));
    });
}

#[divan::bench]
#[ignore]
fn local_acc(bencher: Bencher) {
    let (a, b) = setup(MAT_SIZE);
    bencher.bench_local(|| {
        black_box(naive_local_mult(&a, &b));
    });
}

#[divan::bench]
fn fused_mul_add_bench(bencher: Bencher) {
    let (a, b) = setup(MAT_SIZE);
    bencher.bench_local(|| {
        black_box(fused_mul_add(&a, &b));
    });
}

#[divan::bench(args = TILE_SIZES)]
#[ignore]
fn tiled(bencher: Bencher, tile_size: usize) {
    let (a, b) = setup(MAT_SIZE);
    bencher.bench_local(|| {
        black_box(tiled_mult(&a, &b, tile_size));
    });
}

#[divan::bench(args = cart_prod(TILE_SIZES, THREAD_COUNTS))]
#[ignore]
fn par_tiled(bencher: Bencher, arg: (usize, usize)) {
    let (tile_size, thread_count) = arg;
    let (a, b) = setup(MAT_SIZE);
    bencher.bench_local(|| {
        black_box(parallel_tiled_mult(&a, &b, tile_size, thread_count));
    });
}
