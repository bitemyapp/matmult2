use super::*;
use approx::assert_abs_diff_eq;

#[test]
fn ordered_matches_reference_bits() {
    for (m, k, n) in [(0, 3, 2), (3, 0, 2), (3, 2, 0), (1, 1, 1),
        (3, 5, 7), (63, 64, 65), (65, 129, 67), (128, 127, 129)] {
        let mut a = FlatMatrix::new(m, k);
        let mut b = FlatMatrix::new(k, n);
        random_fill(&mut a, 1);
        random_fill(&mut b, 2);
        for x in &mut a.vals { *x = *x * 2.0 - 1.0; }
        for x in &mut b.vals { *x = *x * 2.0 - 1.0; }
        let reference = naive_local_mult(&a, &b);
        let actual = ordered_mult(&a, &b);
        assert_eq!((actual.rows, actual.cols), (m, n));
        assert_eq!(reference.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                   actual.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
}

#[test]
fn fast_backends_match_rectangular_reference() {
    for threads in [1, 2, 4] {
        let engine = CpuMultiplier::new(threads);
        for (m, k, n) in [(0, 3, 2), (3, 0, 2), (3, 2, 0), (1, 1, 1),
            (3, 5, 7), (63, 64, 65), (65, 129, 67), (128, 127, 129), (7, 513, 19)] {
            let mut a = FlatMatrix::new(m, k);
            let mut b = FlatMatrix::new(k, n);
            random_fill(&mut a, 1);
            random_fill(&mut b, 2);
            for x in &mut a.vals { *x = *x * 2.0 - 1.0; }
            for x in &mut b.vals { *x = *x * 2.0 - 1.0; }
            let reference = naive_local_mult(&a, &b);
            assert_approx_eq(&reference, &engine.multiply(&a, &b), 1e-11);
            #[cfg(target_os = "macos")]
            assert_approx_eq(&reference, &accelerate_mult(&a, &b), 1e-11);
        }
    }
}

fn assert_approx_eq(a: &FlatMatrix, b: &FlatMatrix, tolerance: f64) {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.vals.len(), b.vals.len());

    for i in 0..(a.vals.len()) {
        assert_abs_diff_eq!(a.vals[i], b.vals[i], epsilon = tolerance);
    }
}

#[test]
fn naive_mult_test() {
    let a = FlatMatrix {
        rows: 3,
        cols: 2,
        vals: vec![1., 2., 3., 4., 5., 6.],
    };
    let b = FlatMatrix {
        rows: 2,
        cols: 2,
        vals: vec![1., 2., 3., 4.],
    };
    let prod = FlatMatrix {
        rows: 3,
        cols: 2,
        vals: vec![7., 10., 15., 22., 23., 34.],
    };

    assert_eq!(naive_mult(&a, &b), prod);
}

#[test]
fn naive_local_test() {
    for rows in 1..5 {
        for cols in 1..5 {
            for inner in 1..5 {
                let mut a = FlatMatrix::new(rows, inner);
                let mut b = FlatMatrix::new(inner, cols);
                random_fill_random_seed(&mut a);
                random_fill_random_seed(&mut b);

                assert_eq!(naive_mult(&a, &b), naive_local_mult(&a, &b));
            }
        }
    }
}

#[test]
fn tiled_test() {
    for rows in 1..10 {
        for cols in 1..10 {
            for inner in 1..10 {
                for tile_size in 1..5 {
                    let mut a = FlatMatrix::new(rows, inner);
                    let mut b = FlatMatrix::new(inner, cols);
                    random_fill_random_seed(&mut a);
                    random_fill_random_seed(&mut b);

                    assert_approx_eq(&naive_mult(&a, &b), &tiled_mult(&a, &b, tile_size), 0.00001);
                }
            }
        }
    }
}

#[test]
fn par_tiled_test() {
    for rows in 1..10 {
        for cols in 1..10 {
            for inner in 1..10 {
                for tile_size in 1..5 {
                    for threads in 1..3 {
                        let mut a = FlatMatrix::new(rows, inner);
                        let mut b = FlatMatrix::new(inner, cols);
                        random_fill_random_seed(&mut a);
                        random_fill_random_seed(&mut b);

                        assert_approx_eq(
                            &naive_mult(&a, &b),
                            &&parallel_tiled_mult(&a, &b, tile_size, threads),
                            0.00001,
                        );
                    }
                }
            }
        }
    }
}
