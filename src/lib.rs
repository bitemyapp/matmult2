use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::cmp::min;
use std::mem::swap;

mod optimized;
pub use optimized::ordered_mult;

#[derive(PartialEq, Debug)]
pub struct FlatMatrix {
    rows: usize,
    cols: usize,
    vals: Vec<f64>,
}

impl FlatMatrix {
    /// Row-major elements, for inspecting results without copying them.
    pub fn as_slice(&self) -> &[f64] {
        &self.vals
    }

    pub fn new(rows: usize, cols: usize) -> FlatMatrix {
        let vals: Vec<f64> = vec![0.0; rows * cols];
        FlatMatrix { rows, cols, vals }
    }

    fn at(&self, row: usize, col: usize) -> f64 {
        self.vals[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, val: f64) {
        self.vals[row * self.cols + col] = val;
    }

    fn print(&self) -> String {
        let mut s = "".to_string();
        for i in 0..self.rows {
            s += "\n";
            for j in 0..self.cols {
                s += &format!("{}, ", self.at(i, j));
            }
        }
        s
    }

    fn transpose(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.vals.swap(i * self.cols + j, j * self.rows + i);
            }
        }
        swap(&mut self.rows, &mut self.cols);
    }
}

pub fn random_fill(m: &mut FlatMatrix, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);

    for val in m.vals.iter_mut() {
        *val = rng.random();
    }
}

fn random_fill_random_seed(m: &mut FlatMatrix) {
    let seed: u64 = rand::random();
    random_fill(m, seed);
}

fn validate_dimensions(m1: &FlatMatrix, m2: &FlatMatrix) {
    if m1.cols != m2.rows {
        panic!(
            "Cannot calculate product of matrices with missmatched dimensions: {} != {}",
            m1.rows, m2.cols
        );
    }
}

pub fn naive_mult(m1: &FlatMatrix, m2: &FlatMatrix) -> FlatMatrix {
    validate_dimensions(m1, m2);

    let mut out = FlatMatrix::new(m1.rows, m2.cols);

    for i in 0..m1.rows {
        for j in 0..m2.cols {
            for k in 0..m1.cols {
                out.set(i, j, out.at(i, j) + m1.at(i, k) * m2.at(k, j));
            }
        }
    }

    out
}

pub fn naive_local_mult(m1: &FlatMatrix, m2: &FlatMatrix) -> FlatMatrix {
    validate_dimensions(m1, m2);

    let mut out = FlatMatrix::new(m1.rows, m2.cols);

    for i in 0..m1.rows {
        for j in 0..m2.cols {
            let mut acc = 0.0;
            for k in 0..m1.cols {
                acc += m1.at(i, k) * m2.at(k, j);
            }
            out.set(i, j, acc);
        }
    }

    out
}

pub fn fused_mul_add(m1: &FlatMatrix, m2: &FlatMatrix) -> FlatMatrix {
    validate_dimensions(m1, m2);

    let mut out = FlatMatrix::new(m1.rows, m2.cols);

    for i in 0..m1.rows {
        for j in 0..m2.cols {
            let mut acc = 0.0;
            for k in 0..m1.cols {
                acc = m1.at(i, k).mul_add(m2.at(k, j), acc);
            }
            out.set(i, j, acc);
        }
    }

    out
}

pub fn tiled_mult(m1: &FlatMatrix, m2: &FlatMatrix, tile_size: usize) -> FlatMatrix {
    validate_dimensions(m1, m2);

    let mut prod = FlatMatrix::new(m1.rows, m2.cols);

    for i in (0..m1.rows).step_by(tile_size) {
        for j in (0..m2.cols).step_by(tile_size) {
            for k in (0..m1.cols).step_by(tile_size) {
                for x in i..min(i + tile_size, prod.rows) {
                    for y in j..min(j + tile_size, prod.cols) {
                        let mut acc = 0.0;
                        for z in k..min(k + tile_size, m1.cols) {
                            acc += m1.at(x, z) * m2.at(z, y);
                        }
                        prod.set(x, y, acc + prod.at(x, y));
                    }
                }
            }
        }
    }

    prod
}

pub fn parallel_tiled_mult(
    m1: &FlatMatrix,
    m2: &FlatMatrix,
    tile_size: usize,
    threads: usize,
) -> FlatMatrix {
    validate_dimensions(m1, m2);
    let rows = m1.rows;
    let cols = m2.cols;

    let mut vals = vec![0.0; rows * cols];

    let tile_rows = rows.div_ceil(tile_size);
    let thread_tile_rows = tile_rows.div_ceil(threads);
    let thread_rows = thread_tile_rows * tile_size;

    {
        let bands: Vec<&mut [f64]> = vals
            .chunks_mut(thread_tile_rows * tile_size * m2.cols)
            .collect();

        crossbeam::scope(|spanner| {
            for (t, band) in bands.into_iter().enumerate() {
                spanner.spawn(move |_| {
                    let start_row = t * thread_rows;
                    let end_row = (t + 1) * thread_rows;

                    for i in (start_row..end_row).step_by(tile_size) {
                        for j in (0..cols).step_by(tile_size) {
                            for k in (0..m1.cols).step_by(tile_size) {
                                for x in i..min(i + tile_size, rows) {
                                    for y in j..min(j + tile_size, cols) {
                                        let mut acc = 0.0;
                                        for z in k..min(k + tile_size, m1.cols) {
                                            acc += m1.at(x, z) * m2.at(z, y);
                                        }
                                        band[(x - start_row) * cols + y] += acc;
                                    }
                                }
                            }
                        }
                    }
                });
            }
        })
        .unwrap();
    }

    FlatMatrix { rows, cols, vals }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
