use super::{FlatMatrix, validate_dimensions};

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
