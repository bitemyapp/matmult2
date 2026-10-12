use super::{FlatMatrix, validate_dimensions};

/// Reusable CPU workers for packed SIMD matrix multiplication. Construct once
/// and reuse: starting a pool is setup, while matrix packing is done per call.
///
/// Uses f64 throughout, with FMA and possibly different summation order. Results
/// need not be bit-identical to `naive_local_mult`. No GPU backend is used.
pub struct CpuMultiplier {
    pool: Option<rayon::ThreadPool>,
    threads: usize,
}

impl CpuMultiplier {
    pub fn new(threads: usize) -> Self {
        assert!(threads > 0, "thread count must be positive");
        Self {
            pool: (threads > 1).then(|| rayon::ThreadPoolBuilder::new()
                .num_threads(threads).build().expect("create CPU worker pool")),
            threads,
        }
    }

    pub fn multiply(&self, a: &FlatMatrix, b: &FlatMatrix) -> FlatMatrix {
        validate_dimensions(a, b);
        let mut out = FlatMatrix::new(a.rows, b.cols);
        if a.rows == 0 || b.cols == 0 || a.cols == 0 { return out; }
        let compute = || {
            // SAFETY: all matrices are contiguous row-major buffers with lengths
            // matching their dimensions. Positive strides address those buffers.
            // Output is a fresh allocation and cannot alias either input. Input
            // references stay valid until gemm and its scoped workers complete.
            unsafe {
                gemm::gemm(a.rows, b.cols, a.cols,
                    out.vals.as_mut_ptr(), 1, b.cols as isize, false,
                    a.vals.as_ptr(), 1, a.cols as isize,
                    b.vals.as_ptr(), 1, b.cols as isize,
                    0.0, 1.0, false, false, false,
                    if self.threads == 1 { gemm::Parallelism::None }
                    else { gemm::Parallelism::Rayon(self.threads) });
            }
        };
        match &self.pool {
            Some(pool) => pool.install(compute),
            None => { let mut compute = compute; compute(); }
        }
        out
    }
}

/// Apple's native CPU BLAS. Accelerate chooses its own CPU thread count.
/// Uses f64, but FMA/reassociation may change rounding versus the naive loop.
#[cfg(target_os = "macos")]
pub fn accelerate_mult(a: &FlatMatrix, b: &FlatMatrix) -> FlatMatrix {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        #[link_name = "cblas_dgemm$NEWLAPACK"]
        fn cblas_dgemm(order: i32, trans_a: i32, trans_b: i32,
            m: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32,
            b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32);
    }
    validate_dimensions(a, b);
    let mut out = FlatMatrix::new(a.rows, b.cols);
    if a.rows == 0 || b.cols == 0 || a.cols == 0 { return out; }
    let m = i32::try_from(a.rows).expect("BLAS rows exceed i32");
    let n = i32::try_from(b.cols).expect("BLAS columns exceed i32");
    let k = i32::try_from(a.cols).expect("BLAS inner dimension exceeds i32");
    // SAFETY: dimensions checked above; row-major strides equal row lengths.
    // Valid immutable inputs and fresh non-aliasing output outlive this call.
    unsafe { cblas_dgemm(101, 111, 111, m, n, k, 1.0,
        a.vals.as_ptr(), k, b.vals.as_ptr(), n, 0.0, out.vals.as_mut_ptr(), n); }
    out
}

/// Cache-blocked, contiguous-row multiplication with the same arithmetic order
/// as `naive_local_mult`. Independent columns can be vectorized without
/// reassociating a dot product or using fused multiply-add.
pub fn ordered_mult(a: &FlatMatrix, b: &FlatMatrix) -> FlatMatrix {
    validate_dimensions(a, b);
    let mut out = FlatMatrix::new(a.rows, b.cols);
    if a.rows == 0 || b.cols == 0 || a.cols == 0 {
        return out;
    }
    const BLOCK: usize = 64;
    for ii in (0..a.rows).step_by(BLOCK) {
        for kk in (0..a.cols).step_by(BLOCK) {
            for jj in (0..b.cols).step_by(BLOCK) {
                let end = (jj + BLOCK).min(b.cols);
                for i in ii..(ii + BLOCK).min(a.rows) {
                    let c = &mut out.vals[i * b.cols + jj..i * b.cols + end];
                    let row = &a.vals[i * a.cols..(i + 1) * a.cols];
                    for (k, &x) in row.iter().enumerate().skip(kk).take(BLOCK) {
                        let rhs = &b.vals[k * b.cols + jj..k * b.cols + end];
                        for (c, &y) in c.iter_mut().zip(rhs) {
                            *c += x * y;
                        }
                    }
                }
            }
        }
    }
    out
}
